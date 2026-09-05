#include "mux_stdio.hpp"

#include "../core/config.hpp"
#include "../engine.hpp"
#include "../io/image.hpp"
#include "../tok/k3_chat1.hpp"
#include "../tok/tokenizer.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <algorithm>
#include <cstdarg>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/select.h>
#include <unistd.h>

namespace mvllm {
namespace {

constexpr size_t kMaxHeader = 4096;
constexpr uint64_t kMaxPayload = 1u << 24;

struct Flight {
    uint64_t id = 0;
    uint64_t sched_id = 0;
    int slot = 0;
    int prompt_tokens = 0;
    int max_tokens = 0;
    int emitted = 0;
    bool cancelled = false;
    std::vector<float> image_rgb;
    int image_w = 0;
    int image_h = 0;
};

struct PendingImage {
    std::vector<float> rgb;
    int w = 0;
    int h = 0;
};

static void emit_line(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
    fflush(stdout);
}

static void emit_stat_n(int n_live) {
    const std::string line = mux_format_stat(n_live);
    fwrite(line.data(), 1, line.size(), stdout);
    fflush(stdout);
}

static void emit_ready() {
    fputs("\x01\x01READY\x01\x01\n", stdout);
    fflush(stdout);
    emit_stat_n(0);
}

static void emit_error(uint64_t id, const char *code) {
    emit_line("ERROR %llu %s\n", static_cast<unsigned long long>(id), code);
}

static void emit_accept(uint64_t id, int prompt_tokens) {
    emit_line("ACCEPT %llu %d\n", static_cast<unsigned long long>(id), prompt_tokens);
}

static void emit_data(uint64_t id, const std::string &piece) {
    emit_line("DATA %llu %zu\n", static_cast<unsigned long long>(id), piece.size());
    if (!piece.empty())
        fwrite(piece.data(), 1, piece.size(), stdout);
    fputc('\n', stdout);
    fflush(stdout);
}

static int stop_kind_of(bool stopped_by_stop, int length_limited) {
    if (stopped_by_stop)
        return 2;
    if (length_limited != 0)
        return 1;
    return 0;
}

static void emit_done(uint64_t id, int emitted, int prompt_tokens, int length_limited,
                      int stop_kind) {
    const std::string line = mux_format_done(id, emitted, prompt_tokens, length_limited, stop_kind);
    fwrite(line.data(), 1, line.size(), stdout);
    fflush(stdout);
}

static bool parse_u64(const std::string &s, uint64_t &v) {
    if (s.empty() || s[0] == '-')
        return false;
    char *end = nullptr;
    errno = 0;
    unsigned long long x = std::strtoull(s.c_str(), &end, 10);
    if (errno || end == s.c_str() || *end)
        return false;
    v = static_cast<uint64_t>(x);
    return true;
}

static bool parse_i32(const std::string &s, int &v) {
    if (s.empty())
        return false;
    char *end = nullptr;
    errno = 0;
    long x = std::strtol(s.c_str(), &end, 10);
    if (errno || end == s.c_str() || *end || x < INT32_MIN || x > INT32_MAX)
        return false;
    v = static_cast<int>(x);
    return true;
}

static bool parse_f32(const std::string &s, float &v) {
    if (s.empty())
        return false;
    char *end = nullptr;
    errno = 0;
    float x = std::strtof(s.c_str(), &end);
    if (errno || end == s.c_str() || *end || !std::isfinite(x))
        return false;
    v = x;
    return true;
}

static void split_ws(const std::string &s, std::vector<std::string> &out) {
    out.clear();
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t'))
            ++i;
        if (i >= s.size())
            break;
        size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != '\t')
            ++j;
        out.emplace_back(s.substr(i, j - i));
        i = j;
    }
}

// 1 = line, 0 = EOF, <0 = partial / oversized.
static int read_line_fd(int fd, std::string &line) {
    line.clear();
    for (;;) {
        char c = 0;
        ssize_t r = ::read(fd, &c, 1);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (r == 0)
            return line.empty() ? 0 : -1;
        if (c == '\n')
            return 1;
        if (c == '\r')
            continue;
        if (line.size() >= kMaxHeader) {
            while (c != '\n') {
                r = ::read(fd, &c, 1);
                if (r <= 0)
                    break;
            }
            return -2;
        }
        line.push_back(c);
    }
}

static bool read_exact(int fd, std::string &buf, size_t n) {
    buf.assign(n, '\0');
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(fd, &buf[got], n - got);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            buf.resize(got);
            return false;
        }
        if (r == 0) {
            buf.resize(got);
            return false;
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

static bool read_trailer(int fd) {
    char c = 0;
    for (;;) {
        ssize_t r = ::read(fd, &c, 1);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (r == 0)
            return false;
        return c == '\n';
    }
}

static bool stdin_ready() {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    int r = ::select(STDIN_FILENO + 1, &rfds, nullptr, nullptr, &tv);
    return r > 0 && FD_ISSET(STDIN_FILENO, &rfds);
}

static bool is_k3chat1(const std::string &payload) {
    return payload.size() >= 8 && payload.compare(0, 8, "K3CHAT1\n") == 0;
}

static int clamp_slot(int slot, int n_slots) {
    if (n_slots < 1)
        n_slots = 1;
    if (slot < 0)
        return 0;
    if (slot >= n_slots)
        return n_slots - 1;
    return slot;
}

static int length_limited_of(const std::vector<int> &toks, int max_tokens, const ModelConfig &cfg,
                             int extra_eos) {
    const int n = static_cast<int>(toks.size());
    if (max_tokens <= 0 || n < max_tokens)
        return 0;
    if (!toks.empty() && is_stop_token(toks.back(), cfg, extra_eos))
        return 0;
    return 1;
}

struct Mux {
    Engine &engine;
    Tokenizer tok;
    bool use_sched = false;
    bool eof = false;
    std::unordered_map<uint64_t, Flight> flights;
    std::unordered_map<uint64_t, uint64_t> sched_to_client;
    // IMAGE stash: keyed by client id; consumed by the next SUBMIT with that id.
    std::unordered_map<uint64_t, PendingImage> pending_images;
    uint64_t generating = 0;

    explicit Mux(Engine &e) : engine(e) {
        std::string terr;
        tok.load(engine.runtime().model_dir, terr);
        use_sched = engine.runtime().kv_slots > 1;
    }

    Flight *find(uint64_t id) {
        auto it = flights.find(id);
        return it == flights.end() ? nullptr : &it->second;
    }

    void emit_stat() { emit_stat_n(static_cast<int>(flights.size())); }

    void forget(uint64_t id) {
        auto it = flights.find(id);
        if (it == flights.end())
            return;
        if (it->second.sched_id)
            sched_to_client.erase(it->second.sched_id);
        flights.erase(it);
        emit_stat();
    }

    bool drain_payload(uint64_t bytes, uint64_t extra, std::string *payload, std::string *extra_out) {
        std::string body;
        std::string ext;
        if (bytes && !read_exact(STDIN_FILENO, body, static_cast<size_t>(bytes))) {
            eof = true;
            return false;
        }
        if (extra && !read_exact(STDIN_FILENO, ext, static_cast<size_t>(extra))) {
            eof = true;
            return false;
        }
        if (!read_trailer(STDIN_FILENO)) {
            eof = true;
            return false;
        }
        if (payload)
            *payload = std::move(body);
        if (extra_out)
            *extra_out = std::move(ext);
        return true;
    }

    void take_pending_image(uint64_t id, Flight &fl, GenParams &gp) {
        auto it = pending_images.find(id);
        if (it == pending_images.end())
            return;
        fl.image_rgb = std::move(it->second.rgb);
        fl.image_w = it->second.w;
        fl.image_h = it->second.h;
        pending_images.erase(it);
        if (!fl.image_rgb.empty() && fl.image_w > 0 && fl.image_h > 0) {
            gp.image_rgb = fl.image_rgb.data();
            gp.image_w = fl.image_w;
            gp.image_h = fl.image_h;
        }
    }

    void on_image(uint64_t id, const std::string &payload, int hint_h, int hint_w) {
        if (id == 0) {
            emit_error(0, "BAD_REQUEST");
            return;
        }
        // In-flight SUBMIT owns this id; do not replace a running job's image.
        if (find(id)) {
            emit_error(id, "DUPLICATE_ID");
            return;
        }
        PendingImage img;
        std::string ierr;
        const uint8_t *data = reinterpret_cast<const uint8_t *>(payload.data());
        Status st = mux_decode_image(data, payload.size(), hint_h, hint_w, img.rgb, img.w, img.h,
                                     ierr);
        if (st != Status::Ok || img.rgb.empty() || img.w < 1 || img.h < 1) {
            emit_error(id, "BAD_FRAME");
            return;
        }
        pending_images[id] = std::move(img);
        emit_accept(id, 0);
    }

    // Consume one stdin command. Returns false on EOF.
    bool handle_command(bool from_generate) {
        std::string line;
        int lr = read_line_fd(STDIN_FILENO, line);
        if (lr == 0) {
            eof = true;
            return false;
        }
        if (lr < 0) {
            emit_error(0, "BAD_FRAME");
            if (lr == -1)
                eof = true;
            return lr != -1;
        }
        std::vector<std::string> f;
        split_ws(line, f);
        if (f.empty())
            return true;

        const std::string &verb = f[0];
        uint64_t id = 0;
        const bool have_id = f.size() >= 2 && parse_u64(f[1], id);

        if (verb == "STOP") {
            if (f.size() != 2 || !have_id || id == 0) {
                emit_error(have_id ? id : 0, "BAD_FRAME");
                return true;
            }
            on_stop(id);
            return true;
        }
        if (verb == "CANCEL") {
            if (f.size() != 2 || !have_id || id == 0) {
                emit_error(have_id ? id : 0, "BAD_FRAME");
                return true;
            }
            on_cancel(id);
            return true;
        }
        if (verb == "IMAGE") {
            uint64_t bytes = 0;
            int gh = 0, gw = 0, img_slot = 0;
            if ((f.size() != 5 && f.size() != 6) || !have_id || !parse_u64(f[2], bytes) ||
                !parse_i32(f[3], gh) || !parse_i32(f[4], gw) ||
                (f.size() == 6 && !parse_i32(f[5], img_slot)) || bytes > kMaxPayload) {
                emit_error(have_id ? id : 0, "BAD_FRAME");
                return true;
            }
            (void)img_slot; // optional; attachment is keyed by id, not slot
            std::string img;
            if (!drain_payload(bytes, 0, &img, nullptr)) {
                emit_error(id, "BAD_FRAME");
                return false;
            }
            on_image(id, img, gh, gw);
            return true;
        }
        if (verb != "SUBMIT") {
            emit_error(have_id ? id : 0, "BAD_FRAME");
            return true;
        }

        int slot = 0, max_tokens = 0;
        uint64_t bytes = 0, extra = 0;
        float temperature = 0.f, top_p = 1.f;
        if (f.size() < 7 || !have_id || !parse_i32(f[2], slot) || !parse_u64(f[3], bytes) ||
            !parse_i32(f[4], max_tokens) || !parse_f32(f[5], temperature) ||
            !parse_f32(f[6], top_p) || (f.size() >= 8 && !parse_u64(f[7], extra))) {
            emit_error(have_id ? id : 0, "BAD_FRAME");
            return true;
        }
        if (bytes > kMaxPayload || extra > kMaxPayload) {
            emit_error(id, "BAD_FRAME");
            return true;
        }

        std::string payload;
        std::string extra_json;
        if (!drain_payload(bytes, extra, &payload, &extra_json)) {
            emit_error(id, "BAD_FRAME");
            return false;
        }
        if (from_generate) {
            if (find(id))
                emit_error(id, "DUPLICATE_ID");
            else
                emit_error(id, "SLOT_BUSY");
            return true;
        }
        on_submit(id, slot, max_tokens, temperature, top_p, payload, extra_json);
        return true;
    }

    void poll_controls(uint64_t active_id) {
        (void)active_id;
        while (!eof && stdin_ready()) {
            if (!handle_command(true))
                break;
        }
    }

    void on_stop(uint64_t id) {
        Flight *fl = find(id);
        if (!fl) {
            emit_error(id, "NOT_FOUND");
            return;
        }
        if (fl->sched_id)
            engine.scheduler().stop(fl->sched_id);
        // generate_ids path cannot abort; DONE is still emitted when it returns.
        if (use_sched)
            harvest();
    }

    void on_cancel(uint64_t id) {
        Flight *fl = find(id);
        if (!fl) {
            emit_error(id, "NOT_FOUND");
            return;
        }
        fl->cancelled = true;
        if (fl->sched_id)
            engine.scheduler().cancel(fl->sched_id);
        if (generating == id)
            return;
        emit_error(id, "CANCELLED");
        forget(id);
    }

    bool tokenize_raw(const std::string &payload, std::vector<int> &ids, std::string &err) {
        ids.clear();
        Status st = tok.encode(payload, ids);
        if (st != Status::Ok) {
            err = "tokenize failed";
            return false;
        }
        return true;
    }

    bool tokenize_k3(const std::string &payload, std::vector<int> &ids, GenParams &gp,
                     std::string &err) {
        K3Chat1 parsed;
        if (!k3_chat1_parse(payload, parsed, err))
            return false;
        gp.think = parsed.think;
        if (!parsed.tools.empty())
            gp.tools = parsed.tools;
        Status st = tok.encode_chat(engine.family(), parsed.msgs, parsed.think, ids, {},
                                    parsed.tools.empty() ? nullptr : &parsed.tools);
        if (st != Status::Ok) {
            err = "tokenize failed";
            return false;
        }
        const ModelConfig &cfg = engine.config();
        if (engine.family() == Family::KimiK3 && cfg.bos >= 0 &&
            (ids.empty() || ids.front() != cfg.bos))
            ids.insert(ids.begin(), cfg.bos);
        return true;
    }

    void on_submit(uint64_t id, int slot, int max_tokens, float temperature, float top_p,
                   const std::string &payload, const std::string &extra) {
        if (id == 0) {
            emit_error(0, "BAD_REQUEST");
            return;
        }
        if (max_tokens < 1) {
            emit_error(id, "BAD_REQUEST");
            return;
        }
        if (find(id)) {
            emit_error(id, "DUPLICATE_ID");
            return;
        }
        slot = clamp_slot(slot, engine.sessions().n_slots());
        if (payload.empty()) {
            emit_error(id, "EMPTY_PROMPT");
            return;
        }

        const bool k3 = is_k3chat1(payload);
        GenParams gp;
        gp.max_new_tokens = max_tokens;
        gp.temperature = temperature;
        gp.top_p = top_p;
        gp.apply_template = false;
        gp.eos = engine.config().eos;
        gp.cache_slot = slot;

        std::vector<int> ids;
        std::string terr;
        bool tok_ok = k3 ? tokenize_k3(payload, ids, gp, terr) : tokenize_raw(payload, ids, terr);
        if (!tok_ok) {
            emit_error(id, "BAD_REQUEST");
            return;
        }
        if (ids.empty()) {
            emit_error(id, "EMPTY_PROMPT");
            return;
        }

        std::string jerr;
        if (!mux_apply_extra_json(extra, gp, jerr)) {
            emit_error(id, "BAD_REQUEST");
            return;
        }
        // persist_path / stop_ids / eos_only stay on gp for generate_ids.
        // Raw prefix_bytes → prefix_reuse tokens; skip K3CHAT1 and explicit reuse.
        if (gp.prefix_bytes > 0 && gp.prefix_reuse == 0 && !k3) {
            const int nb = std::min(gp.prefix_bytes, static_cast<int>(payload.size()));
            std::vector<int> pids;
            std::string perr;
            if (tokenize_raw(payload.substr(0, static_cast<size_t>(nb)), pids, perr) &&
                !pids.empty())
                gp.prefix_reuse =
                    std::min(static_cast<int>(pids.size()), static_cast<int>(ids.size()));
        }
        slot = clamp_slot(gp.cache_slot, engine.sessions().n_slots());
        gp.cache_slot = slot;
        if (gp.max_new_tokens > 0)
            max_tokens = gp.max_new_tokens;

        const int prompt_tokens = static_cast<int>(ids.size());
        Flight fl;
        fl.id = id;
        fl.slot = slot;
        fl.prompt_tokens = prompt_tokens;
        fl.max_tokens = max_tokens;

        const bool sched = use_sched && !k3;
        if (sched) {
            if (engine.sessions().busy(slot)) {
                emit_error(id, "SLOT_BUSY");
                return;
            }
            take_pending_image(id, fl, gp);
            Engine *eng = &engine;
            gp.on_token = [eng, id](int tok) { emit_data(id, eng->decode_token(tok)); };
            gp.token_text = [eng](int tid) { return eng->decode_token(tid); };
            std::string serr;
            uint64_t sid = engine.scheduler().submit(slot, ids, gp, serr);
            if (sid == 0) {
                if (!fl.image_rgb.empty()) {
                    PendingImage back;
                    back.rgb = std::move(fl.image_rgb);
                    back.w = fl.image_w;
                    back.h = fl.image_h;
                    pending_images[id] = std::move(back);
                }
                emit_error(id, serr == "SLOT_BUSY" ? "SLOT_BUSY" : "BAD_REQUEST");
                return;
            }
            emit_accept(id, prompt_tokens);
            fl.sched_id = sid;
            flights[id] = std::move(fl);
            sched_to_client[sid] = id;
            emit_stat();
            return;
        }

        if (!engine.sessions().try_acquire(slot)) {
            emit_error(id, "SLOT_BUSY");
            return;
        }
        take_pending_image(id, fl, gp);
        emit_accept(id, prompt_tokens);
        flights[id] = std::move(fl);
        emit_stat();
        if (Flight *stored = find(id)) {
            if (!stored->image_rgb.empty()) {
                gp.image_rgb = stored->image_rgb.data();
                gp.image_w = stored->image_w;
                gp.image_h = stored->image_h;
            }
        }
        generating = id;

        Engine *eng = &engine;
        Mux *self = this;
        gp.on_token = [eng, self, id](int tok) {
            Flight *live = self->find(id);
            if (!live || live->cancelled)
                return;
            emit_data(id, eng->decode_token(tok));
            ++live->emitted;
            self->poll_controls(id);
        };
        gp.token_text = [eng](int tid) { return eng->decode_token(tid); };

        GenResult out;
        std::string gerr;
        Status st = Status::Ok;
        if (k3)
            st = engine.generate(payload, gp, out, gerr);
        else
            st = engine.generate_ids(ids, gp, out, gerr);
        engine.sessions().release(slot);
        generating = 0;

        Flight *live = find(id);
        const bool cancelled = live && live->cancelled;
        if (cancelled) {
            emit_error(id, "CANCELLED");
            forget(id);
            return;
        }
        if (st != Status::Ok) {
            emit_error(id, "BAD_REQUEST");
            forget(id);
            return;
        }
        const int emitted = live ? live->emitted : static_cast<int>(out.tokens.size());
        const int limited = length_limited_of(out.tokens, max_tokens, engine.config(), gp.eos);
        emit_done(id, emitted, prompt_tokens, limited, stop_kind_of(out.stopped_by_stop, limited));
        forget(id);
    }

    void harvest() {
        std::vector<uint64_t> done;
        for (const auto &kv : flights) {
            const Flight &fl = kv.second;
            if (!fl.sched_id)
                continue;
            if (!engine.scheduler().finished(fl.sched_id))
                continue;
            done.push_back(fl.id);
        }
        for (uint64_t id : done) {
            Flight *fl = find(id);
            if (!fl)
                continue;
            const BatchJob *job = engine.scheduler().job(fl->sched_id);
            if (!job || job->state == BatchJobState::Cancelled) {
                emit_error(fl->id, "CANCELLED");
            } else if (job->status != Status::Ok) {
                emit_error(fl->id, "BAD_REQUEST");
            } else {
                const int emitted = static_cast<int>(job->out.tokens.size());
                const int limited = length_limited_of(job->out.tokens, fl->max_tokens,
                                                      engine.config(), job->gp.eos);
                emit_done(fl->id, emitted, fl->prompt_tokens, limited,
                          stop_kind_of(job->out.stopped_by_stop, limited));
            }
            forget(id);
        }
    }

    void pump() {
        if (!use_sched || flights.empty())
            return;
        engine.scheduler().pump();
        harvest();
    }

    Status run(std::string &err) {
        err.clear();
        setvbuf(stdout, nullptr, _IONBF, 0);
        emit_ready();
        while (!eof || !flights.empty()) {
            if (!eof) {
                if (!use_sched || flights.empty()) {
                    if (!handle_command(false))
                        eof = true;
                } else if (stdin_ready()) {
                    if (!handle_command(false))
                        eof = true;
                }
            }
            pump();
            if (eof && !flights.empty() && use_sched)
                continue;
            if (eof && flights.empty())
                break;
        }
        return Status::Ok;
    }
};

} // namespace

Status mux_stdio_run(Engine &engine, std::string &err) {
    Mux mux(engine);
    return mux.run(err);
}

bool mux_apply_extra_json(const std::string &extra, GenParams &gp, std::string &err) {
    err.clear();
    if (extra.empty())
        return true;

    using json = nlohmann::json;
    json j = json::parse(extra, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        err = "invalid extra json";
        return false;
    }

    auto as_float = [](const json &v, float &out) -> bool {
        if (!v.is_number())
            return false;
        if (v.is_number_float())
            out = static_cast<float>(v.get<double>());
        else if (v.is_number_unsigned())
            out = static_cast<float>(v.get<uint64_t>());
        else
            out = static_cast<float>(v.get<int64_t>());
        return std::isfinite(out);
    };
    auto as_int = [](const json &v, int &out) -> bool {
        if (!v.is_number_integer())
            return false;
        if (v.is_number_unsigned()) {
            const uint64_t u = v.get<uint64_t>();
            if (u > static_cast<uint64_t>(INT32_MAX))
                return false;
            out = static_cast<int>(u);
            return true;
        }
        const int64_t s = v.get<int64_t>();
        if (s < INT32_MIN || s > INT32_MAX)
            return false;
        out = static_cast<int>(s);
        return true;
    };

    if (j.contains("stop")) {
        const json &st = j["stop"];
        if (st.is_string()) {
            gp.stop.clear();
            gp.stop.push_back(st.get<std::string>());
        } else if (st.is_array()) {
            gp.stop.clear();
            for (const auto &s : st) {
                if (s.is_string())
                    gp.stop.push_back(s.get<std::string>());
            }
        }
    }
    if (j.contains("grammar") && j["grammar"].is_string())
        gp.grammar = j["grammar"].get<std::string>();
    if (j.contains("seed") && j["seed"].is_number_integer()) {
        if (j["seed"].is_number_unsigned())
            gp.seed = j["seed"].get<uint64_t>();
        else {
            const int64_t s = j["seed"].get<int64_t>();
            if (s >= 0)
                gp.seed = static_cast<uint64_t>(s);
        }
    }

    float f = 0.f;
    if (j.contains("frequency_penalty") && as_float(j["frequency_penalty"], f))
        gp.frequency_penalty = f;
    if (j.contains("presence_penalty") && as_float(j["presence_penalty"], f))
        gp.presence_penalty = f;
    if (j.contains("repetition_penalty") && as_float(j["repetition_penalty"], f))
        gp.repetition_penalty = f;
    if (j.contains("min_p") && as_float(j["min_p"], f))
        gp.min_p = f;
    if (j.contains("temperature") && as_float(j["temperature"], f))
        gp.temperature = f;
    if (j.contains("top_p") && as_float(j["top_p"], f))
        gp.top_p = f;

    int iv = 0;
    if (j.contains("top_k") && as_int(j["top_k"], iv))
        gp.top_k = iv;
    if (j.contains("logprobs") && as_int(j["logprobs"], iv))
        gp.logprobs = iv;
    if (j.contains("max_tokens") && as_int(j["max_tokens"], iv) && iv > 0)
        gp.max_new_tokens = iv;
    if (j.contains("max_new_tokens") && as_int(j["max_new_tokens"], iv) && iv > 0)
        gp.max_new_tokens = iv;

    if (j.contains("logit_bias") && j["logit_bias"].is_object()) {
        gp.logit_bias.clear();
        for (auto it = j["logit_bias"].begin(); it != j["logit_bias"].end(); ++it) {
            char *end = nullptr;
            errno = 0;
            const long tid = std::strtol(it.key().c_str(), &end, 10);
            if (errno || !end || end == it.key().c_str() || *end || tid < INT32_MIN ||
                tid > INT32_MAX)
                continue;
            float bias = 0.f;
            if (!as_float(it.value(), bias))
                continue;
            gp.logit_bias.emplace_back(static_cast<int>(tid), bias);
        }
    }

    if (j.contains("cache_slot") && as_int(j["cache_slot"], iv))
        gp.cache_slot = iv;
    if (j.contains("think") && j["think"].is_boolean())
        gp.think = j["think"].get<bool>();
    if (j.contains("enable_thinking") && j["enable_thinking"].is_boolean())
        gp.think = j["enable_thinking"].get<bool>();

    auto apply_persist_path = [&](const char *key) -> bool {
        if (!j.contains(key))
            return true;
        if (!j[key].is_string()) {
            err = "invalid extra json";
            return false;
        }
        gp.persist_path = j[key].get<std::string>();
        return true;
    };
    if (!apply_persist_path("persist") || !apply_persist_path("kv_path") ||
        !apply_persist_path("coli_kv"))
        return false;

    auto apply_persist_ver = [&](const char *key) -> bool {
        if (!j.contains(key))
            return true;
        if (!as_int(j[key], iv) || iv < 1 || iv > 3) {
            err = "invalid extra json";
            return false;
        }
        gp.persist_ver = iv;
        return true;
    };
    if (!apply_persist_ver("persist_ver") || !apply_persist_ver("kv_ver"))
        return false;

    if (j.contains("prefix_bytes")) {
        if (!as_int(j["prefix_bytes"], iv) || iv < 0) {
            err = "invalid extra json";
            return false;
        }
        gp.prefix_bytes = iv;
    }
    if (j.contains("prefix_reuse")) {
        if (!as_int(j["prefix_reuse"], iv) || iv < 0) {
            err = "invalid extra json";
            return false;
        }
        gp.prefix_reuse = iv;
    }

    if (j.contains("stop_ids")) {
        const json &arr = j["stop_ids"];
        if (!arr.is_array()) {
            err = "invalid extra json";
            return false;
        }
        gp.stop_ids.clear();
        for (const auto &x : arr) {
            if (as_int(x, iv))
                gp.stop_ids.push_back(iv);
        }
    }
    if (j.contains("eos_only")) {
        if (!j["eos_only"].is_boolean()) {
            err = "invalid extra json";
            return false;
        }
        gp.eos_only = j["eos_only"].get<bool>();
    }

    return true;
}

Status mux_decode_image(const uint8_t *data, size_t n, int hint_h, int hint_w,
                        std::vector<float> &rgb, int &width, int &height, std::string &err) {
    rgb.clear();
    width = 0;
    height = 0;
    err.clear();

    if (data && n > 0) {
        const Status st = decode_image_bytes(data, n, rgb, width, height, err);
        if (st == Status::Ok)
            return Status::Ok;
    }
    rgb.clear();
    width = 0;
    height = 0;

    if (data && hint_h > 0 && hint_w > 0) {
        const uint64_t hw = static_cast<uint64_t>(hint_h) * static_cast<uint64_t>(hint_w);
        if (hw <= UINT64_MAX / 3ull && hw * 3ull == static_cast<uint64_t>(n)) {
            rgb.resize(n);
            for (size_t i = 0; i < n; ++i)
                rgb[i] = static_cast<float>(data[i]) * (1.f / 255.f);
            width = hint_w;
            height = hint_h;
            err.clear();
            return Status::Ok;
        }
    }

    if (err.empty())
        err = "bad image";
    return Status::ParseError;
}

std::string mux_format_done(uint64_t id, int emitted, int prompt_tokens, int length_limited,
                            int stop_kind) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "DONE %llu STAT %d 0.00 0.0 0.00 %d %d %d\n",
                  static_cast<unsigned long long>(id), emitted, prompt_tokens, length_limited,
                  stop_kind);
    return buf;
}

std::string mux_format_stat(int n_live, double tps, double tpot, double load) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "STAT %d %.2f %.1f %.2f\n", n_live, tps, tpot, load);
    return buf;
}

} // namespace mvllm
