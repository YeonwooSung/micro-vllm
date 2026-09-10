#include "core/config.hpp"
#include "engine.hpp"
#include "gpu/backend.hpp"
#include "gpu/coli_cuda.hpp"
#include "gpu/dsv4_cuda.hpp"
#include "gpu/metal_ops.hpp"
#include "gpu/metal_h3.hpp"
#include "gpu/vk_ops.hpp"
#include "legacy/llama_dims.hpp"
#include "io/dump_env.hpp"
#include "io/file_io.hpp"
#include "io/safetensors.hpp"
#include "io/shard_probe.hpp"
#include "model/family.hpp"
#include "model/h3_vae.hpp"
#include "quant/quant.hpp"
#include "quant/weight.hpp"
#include "serve/http_server.hpp"
#include "io/image.hpp"
#include "io/av_mux.hpp"
#include "io/h3_resize.hpp"
#include "quant/kv_fp8.hpp"
#include "quant/kv_tq.hpp"
#include "model/h3_text.hpp"
#include "model/h3_audio_vae.hpp"
#include "model/h3_layout.hpp"
#include "model/h3_seg.hpp"
#include "model/h3_vision.hpp"
#include "model/h3_mm.hpp"
#include "model/h3_dit_schedule.hpp"
#include "model/h3_canvas.hpp"
#include "model/h3_adaln.hpp"
#include "model/h3_reuse.hpp"
#include "model/h3_token_reduce.hpp"
#include "model/moe_pick.hpp"
#include "store/block_store.hpp"
#include "store/expert_store.hpp"
#include "store/kv_persist.hpp"
#include "store/kv_persist_v2.hpp"
#include "store/kv_persist_v3.hpp"
#include "store/route_usage.hpp"
#include "store/route_trace.hpp"
#include "store/kv_prefix.hpp"
#include "store/kv_row.hpp"
#include "quant/int8_dyn.hpp"
#include "quant/native_act.hpp"
#include "quant/native_fp4.hpp"
#include "quant/native_batch.hpp"
#include "quant/bf16.hpp"
#include "tok/tokenizer.hpp"
#include "tok/k3_tools.hpp"
#include "tok/gbnf.hpp"
#include "tok/gbnf_forced.hpp"
#include "tok/sample_nuc.hpp"
#include "tok/stop_set.hpp"
#include "tok/logprob.hpp"
#include "tok/utf8.hpp"
#include "tok/logit_dump.hpp"
#include "tok/json_schema.hpp"
#include "tok/k3_chat1.hpp"
#include "tok/decode_post.hpp"
#include "tok/glm_tools.hpp"
#include "serve/session.hpp"
#include "serve/scheduler.hpp"
#include "serve/anthropic.hpp"
#include "serve/mux_stdio.hpp"
#include "serve/mux_frames.hpp"
#include "serve/mux_codec.hpp"
#include "serve/mux_submit.hpp"
#include "serve/legacy_stdio.hpp"
#include "serve/hwinfo.hpp"
#include "serve/cli_flags.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << #cond << "\n";          \
            ++g_fail;                                                                              \
        } else {                                                                                   \
            ++g_pass;                                                                              \
        }                                                                              \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                                      \
    do {                                                                                           \
        double _aa = (double)(a), _bb = (double)(b);                                               \
        if (std::fabs(_aa - _bb) > (eps)) {                                                        \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " << _aa << " !~ " << _bb    \
                      << "\n";                                                                     \
            ++g_fail;                                                                              \
        } else {                                                                                   \
            ++g_pass;                                                                              \
        }                                                                              \
    } while (0)

static std::string tmpdir() {
    char buf[] = "/tmp/mvllmXXXXXX";
    if (!mkdtemp(buf)) {
        std::cerr << "mkdtemp failed\n";
        std::exit(1);
    }
    return buf;
}

static void write_file(const std::string &path, const std::string &s) {
    std::ofstream o(path);
    o << s;
}

static void test_quant() {
    using namespace mvllm::quant;
    // matmul identity
    float w[4] = {1, 0, 0, 1};
    float x[2] = {3, 4};
    float y[2] = {0, 0};
    matmul_f32(y, x, w, 1, 2, 2);
    CHECK_NEAR(y[0], 3, 1e-5);
    CHECK_NEAR(y[1], 4, 1e-5);

    // SiTU at 0 is 0
    CHECK_NEAR(situ_glu(0, 0, 4, 25), 0, 1e-6);
    // clamped swiglu: large gate saturates
    float a = clamped_swiglu(100.f, 1.f, 7.f);
    float b = 7.f * sigmoid(7.f) * 1.f;
    CHECK_NEAR(a, b, 1e-4);

    // int4 roundtrip matmul vs f32 on a tiny well-scaled matrix
    const int O = 4, I = 64;
    std::vector<float> W(O * I), X(I), Yf(O), Yq(O);
    for (int i = 0; i < O * I; ++i)
        W[i] = ((i * 17) % 11 - 5) * 0.1f;
    for (int i = 0; i < I; ++i)
        X[i] = ((i * 3) % 7 - 3) * 0.2f;
    matmul_f32(Yf.data(), X.data(), W.data(), 1, I, O);
    std::vector<uint8_t> packed((O * I + 1) / 2);
    std::vector<float> scales(O * ((I + 63) / 64));
    quantize_int4_g64(W.data(), O, I, packed.data(), scales.data());
    matmul_int4_g64(Yq.data(), X.data(), packed.data(), scales.data(), 1, I, O);
    for (int o = 0; o < O; ++o)
        CHECK(std::fabs(Yf[o] - Yq[o]) < 0.6f); // 4-bit, loose

    // mxfp4
    std::vector<uint8_t> mp((O * I + 1) / 2), ms(O * ((I + 31) / 32));
    pack_mxfp4(W.data(), O, I, mp.data(), ms.data());
    std::vector<float> Ym(O);
    matmul_mxfp4(Ym.data(), X.data(), mp.data(), ms.data(), 1, I, O);
    for (int o = 0; o < O; ++o)
        CHECK(std::isfinite(Ym[o]));

    // rmsnorm unit
    float xn[4] = {3, 0, 0, 4};
    float wn[4] = {1, 1, 1, 1};
    float yn[4];
    rmsnorm(xn, wn, yn, 4, 0.f);
    float mean = (9 + 16) / 4.f;
    CHECK_NEAR(yn[0], 3.f / std::sqrt(mean), 1e-5);
    CHECK_NEAR(yn[3], 4.f / std::sqrt(mean), 1e-5);

    float sm[3] = {0, 0, 0};
    softmax_inplace(sm, 3);
    CHECK_NEAR(sm[0], 1.f / 3.f, 1e-5);
}

static void test_quant_mat() {
    using namespace mvllm::quant;
    CHECK(sanitize_bits(3) == 4);
    CHECK(sanitize_bits(4) == 4);
    CHECK(sanitize_bits(5) == 8);
    CHECK(sanitize_bits(8) == 8);
    CHECK(sanitize_bits(16) == 32);
    CHECK(sanitize_bits(32) == 32);

    const int O = 4, I = 64;
    std::vector<float> W(static_cast<size_t>(O) * I), X(I), Yf(O), Yq(O);
    for (int i = 0; i < O * I; ++i)
        W[i] = ((i * 17) % 11 - 5) * 0.1f;
    for (int i = 0; i < I; ++i)
        X[i] = ((i * 3) % 7 - 3) * 0.2f;
    matmul_f32(Yf.data(), X.data(), W.data(), 1, I, O);

    QuantMat f32;
    f32.from_f32(W.data(), O, I, 32);
    CHECK(f32.fmt == 0);
    CHECK(f32.O == O && f32.I == I);
    std::vector<float> Y32(O);
    f32.gemm(Y32.data(), X.data(), 1);
    for (int o = 0; o < O; ++o)
        CHECK_NEAR(Y32[o], Yf[o], 1e-5);
    CHECK(f32.bytes() == static_cast<int64_t>(O) * I * static_cast<int64_t>(sizeof(float)));

    QuantMat i8;
    i8.from_f32(W.data(), O, I, 8);
    CHECK(i8.fmt == 8);
    CHECK(i8.q8.size() == static_cast<size_t>(O) * I);
    CHECK(i8.scales.size() == static_cast<size_t>(O));
    i8.gemm(Yq.data(), X.data(), 1);
    for (int o = 0; o < O; ++o)
        CHECK(std::fabs(Yf[o] - Yq[o]) < 0.35f);
    CHECK(i8.bytes() < f32.bytes());

    QuantMat i4;
    i4.from_f32(W.data(), O, I, 4);
    CHECK(i4.fmt == 4);
    CHECK(i4.q4.size() == static_cast<size_t>(O) * (I / 2));
    CHECK(i4.scales.size() == static_cast<size_t>(O) * (I / 64));
    std::fill(Yq.begin(), Yq.end(), 0.f);
    i4.gemm(Yq.data(), X.data(), 1);
    for (int o = 0; o < O; ++o)
        CHECK(std::fabs(Yf[o] - Yq[o]) < 0.6f);
    CHECK(i4.bytes() < i8.bytes());

    // bits<=4 but I not divisible by 64 → int8 fallback
    const int Iodd = 40;
    std::vector<float> Wodd(static_cast<size_t>(O) * Iodd, 0.1f);
    QuantMat fallback;
    fallback.from_f32(Wodd.data(), O, Iodd, 4);
    CHECK(fallback.fmt == 8);
    CHECK(fallback.q8.size() == static_cast<size_t>(O) * Iodd);

    // Batch S=2
    std::vector<float> X2(static_cast<size_t>(2) * I), Yb(static_cast<size_t>(2) * O);
    for (int i = 0; i < 2 * I; ++i)
        X2[i] = ((i * 5) % 9 - 4) * 0.15f;
    i4.gemm(Yb.data(), X2.data(), 2);
    for (int s = 0; s < 2; ++s)
        for (int o = 0; o < O; ++o)
            CHECK(std::isfinite(Yb[s * O + o]));
}

static void test_expert_store() {
    using namespace mvllm;
    std::string dir = tmpdir();
    const int nL = 2, nE = 4;
    const int64_t ebytes = 4096;
    std::string path = dir + "/experts.bin";
    {
        std::ofstream o(path, std::ios::binary);
        std::vector<char> buf(ebytes);
        for (int l = 0; l < nL; ++l)
            for (int e = 0; e < nE; ++e) {
                std::memset(buf.data(), 0, ebytes);
                buf[0] = static_cast<char>(l);
                buf[1] = static_cast<char>(e);
                o.write(buf.data(), ebytes);
            }
    }
    ExpertStore store;
    std::string err;
    // capacity = 2 experts total across 2 layers => 1 slot/layer
    CHECK(store.open(nL, nE, ebytes, ebytes * 2, err) == Status::Ok);
    int64_t off = 0;
    for (int l = 0; l < nL; ++l)
        for (int e = 0; e < nE; ++e) {
            ExpertLoc loc;
            loc.key = {l, e};
            loc.path = path;
            loc.offset = off;
            loc.bytes = ebytes;
            CHECK(store.register_expert(loc, err) == Status::Ok);
            off += ebytes;
        }
    ExpertView v{};
    CHECK(store.lookup({0, 1}, v, err) == Status::Ok);
    CHECK(v.data && v.data[0] == 0 && v.data[1] == 1);
    store.release(v);
    CHECK(store.lookup({0, 1}, v, err) == Status::Ok); // hit
    store.release(v);
    ExpertStoreStats st{};
    store.stats(st);
    CHECK(st.hits >= 1);
    CHECK(st.misses >= 1);
    // evict
    CHECK(store.lookup({0, 2}, v, err) == Status::Ok);
    CHECK(v.data[1] == 2);
    store.release(v);
    ExpertKey keys[2] = {{1, 0}, {1, 3}};
    CHECK(store.prefetch(keys, 2, err) == Status::Ok);
    CHECK(store.resident(0, 2));
    CHECK(!store.resident(0, 1));
    {
        RepinEvent ev[4];
        int nr = store.take_repin(ev, 4);
        CHECK(nr >= 1);
        CHECK(ev[0].layer == 0 && ev[0].eid == 1 && ev[0].old_tier == 1 && ev[0].gpu == 0);
        CHECK(store.take_repin(ev, 4) == 0);
    }
    CHECK(!store.resident(-1, 0));
    int tiers[8];
    CHECK(store.fill_tiers(tiers, 8) == 8);
    CHECK(tiers[2] == 1); // layer 0 expert 2
    CHECK(tiers[1] == 0); // evicted
    int ram = 0, disk = 0;
    store.count_tiers(ram, disk);
    CHECK(ram >= 1 && ram + disk == nL * nE);
    CHECK(store.fill_tiers(nullptr, 8) == 0);
    double edisk = 0, ewait = 0;
    store.io_perf(edisk, ewait);
    CHECK(edisk > 0.0);
    CHECK(ewait > 0.0);
    store.take_io_perf(edisk, ewait, true);
    store.io_perf(edisk, ewait);
    CHECK(edisk == 0.0 && ewait == 0.0);
    store.close();
    CHECK(!store.resident(0, 2));
    store.count_tiers(ram, disk);
    CHECK(ram == 0 && disk == 0);

    // 8 GiB / 4 KiB / 2 layers would be ~1M slots without the n_experts cap.
    CHECK(expert_store_slots_per_layer(2, 4, 4096, 8LL * 1024 * 1024 * 1024) == 4);
    CHECK(expert_store_slots_per_layer(2, 4, 4096, 4096) == 1);
    CHECK(expert_store_slots_per_layer(93, 896, 14LL * 1024 * 1024, 8LL * 1024 * 1024 * 1024) > 0);
    CHECK(expert_store_slots_per_layer(93, 896, 14LL * 1024 * 1024, 8LL * 1024 * 1024 * 1024) <=
          896);
}

static void test_block_store() {
    using namespace mvllm;
    std::string dir = tmpdir();
    std::string path = dir + "/blocks.bin";
    const int64_t bsz = 2048;
    {
        std::ofstream o(path, std::ios::binary);
        std::vector<char> buf(bsz);
        for (int i = 0; i < 4; ++i) {
            std::memset(buf.data(), (char)i, bsz);
            o.write(buf.data(), bsz);
        }
    }
    BlockStore bs;
    std::string err;
    CHECK(bs.open(4, bsz, 2, err) == Status::Ok);
    for (int i = 0; i < 4; ++i)
        CHECK(bs.register_block(i, path, i * bsz, bsz, err) == Status::Ok);
    const uint8_t *p = nullptr;
    int64_t n = 0;
    CHECK(bs.acquire(0, &p, &n, err) == Status::Ok);
    CHECK(p && p[0] == 0 && n == bsz);
    CHECK(bs.prefetch(1, err) == Status::Ok);
    const uint8_t *p1 = nullptr;
    CHECK(bs.acquire(1, &p1, &n, err) == Status::Ok);
    CHECK(p1 && p1[0] == 1);
    bs.release(0);
    bs.release(1);
    CHECK(bs.hits() + bs.misses() >= 2);
    bs.close();
}

static void test_safetensors() {
    using namespace mvllm;
    using namespace mvllm::io;
    std::string dir = tmpdir();
    std::string path = dir + "/t.safetensors";
    // header: {"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}
    std::string header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    uint64_t hs = header.size();
    float data[2] = {1.5f, -2.5f};
    {
        std::ofstream o(path, std::ios::binary);
        o.write(reinterpret_cast<const char *>(&hs), 8);
        o.write(header.data(), header.size());
        o.write(reinterpret_cast<const char *>(data), 8);
    }
    StFile f;
    std::string err;
    CHECK(st_open(path, f, err) == Status::Ok);
    const StTensor *t = st_find(f, "a");
    CHECK(t && t->shape.size() == 1 && t->shape[0] == 2);
    float got[2] = {0, 0};
    CHECK(st_read(f, *t, got, err) == Status::Ok);
    CHECK_NEAR(got[0], 1.5f, 1e-6);
    CHECK_NEAR(got[1], -2.5f, 1e-6);
    st_close(f);
}

static void test_http_helpers() {
    using namespace mvllm;
    std::string s = json_escape("a\"b\\c");
    CHECK(s.find("\\\"") != std::string::npos);
    std::string chat = openai_chat_response("id1", "kimi", "hello", 3, 2);
    CHECK(chat.find("\"object\":\"chat.completion\"") != std::string::npos ||
          chat.find("chat.completion") != std::string::npos);
    CHECK(chat.find("hello") != std::string::npos);
    CHECK(chat.find("reasoning_content") == std::string::npos);
    std::string chatr = openai_chat_response("id1", "kimi", "hello", 3, 2, "I think");
    CHECK(chatr.find("reasoning_content") != std::string::npos);
    CHECK(chatr.find("I think") != std::string::npos);
    std::string tc;
    CHECK(extract_tool_choice("{\"tool_choice\":\"none\"}", tc) && tc == "none");
    CHECK(extract_tool_choice(
        "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}}", tc) &&
          tc == "get_weather");
    std::vector<std::string> stops;
    CHECK(extract_json_string_array("{\"stop\":[\"END\",\"STOP\"]}", "stop", stops) &&
          stops.size() == 2);
    std::string models = openai_models_response("glm53");
    CHECK(models.find("glm53") != std::string::npos);
    std::string body = "{\"prompt\":\"hi\",\"max_tokens\":8}";
    std::string p;
    int n = 0;
    CHECK(extract_json_string(body, "prompt", p) && p == "hi");
    CHECK(extract_json_int(body, "max_tokens", n) && n == 8);
    std::vector<ChatMessage> msgs;
    CHECK(extract_chat_messages(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hello\"},{\"role\":\"assistant\",\"content\":\"ok\"}]}",
        msgs));
    CHECK(msgs.size() == 2 && msgs[0].role == "user" && msgs[0].content == "hello");
    bool th = false;
    CHECK(extract_json_bool("{\"enable_thinking\":true}", "enable_thinking", th) && th);
    CHECK(extract_json_bool("{\"enable_thinking\":false}", "enable_thinking", th) && !th);
    float temp = 0.f, topp = 0.f;
    CHECK(extract_json_number("{\"temperature\":0.7,\"top_p\":0.9}", "temperature", temp) &&
          std::fabs(temp - 0.7f) < 1e-5f);
    CHECK(extract_json_number("{\"temperature\":0.7,\"top_p\":0.9}", "top_p", topp) &&
          std::fabs(topp - 0.9f) < 1e-5f);
    std::string sse = openai_sse_chunk("id1", "kimi", "{\"content\":\"hi\"}", nullptr);
    CHECK(sse.find("data: ") == 0);
    CHECK(sse.find("chat.completion.chunk") != std::string::npos);
    CHECK(sse.find("\"content\":\"hi\"") != std::string::npos);
    CHECK(sse.find("finish_reason\":null") != std::string::npos);
    CHECK(sse.size() >= 2 && sse.substr(sse.size() - 2) == "\n\n");
    std::string ssed = openai_sse_chunk("id1", "kimi", "{}", "stop");
    CHECK(ssed.find("\"finish_reason\":\"stop\"") != std::string::npos);
    CHECK(openai_sse_done() == "data: [DONE]\n\n");
    std::vector<ChatMessage> tools;
    CHECK(extract_chat_messages(
        "{\"messages\":[{\"role\":\"tool\",\"content\":\"sunny\",\"name\":\"get_weather\"}]}",
        tools));
    CHECK(tools.size() == 1 && tools[0].role == "tool" && tools[0].tool_name == "get_weather");
    std::vector<ChatMessage> imgs;
    CHECK(extract_chat_messages(
        "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"see \"},"
        "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,QQ==\"}}]}]}",
        imgs));
    CHECK(imgs.size() == 1 && imgs[0].content == "see " && imgs[0].image_urls.size() == 1);
    CHECK(imgs[0].image_urls[0].find("data:") == 0);

    int rh = 0, rw = 0;
    glm_smart_resize(480, 640, &rh, &rw, 14, 2, 2);
    CHECK(rh % 28 == 0 && rw % 28 == 0);
    CHECK(rh > 0 && rw > 0);

    const char ppm[] = "P6\n2 2\n255\n"
                       "\xff\x00\x00\x00\xff\x00\x00\x00\xff\xff\xff\x00";
    std::vector<uint8_t> raw(ppm, ppm + sizeof(ppm) - 1);
    std::vector<float> rgb;
    int iw = 0, ih = 0;
    std::string ierr;
    CHECK(decode_image_bytes(raw.data(), raw.size(), rgb, iw, ih, ierr) == Status::Ok);
    CHECK(iw == 2 && ih == 2 && rgb.size() == 12);
    CHECK(rgb[0] > 0.9f && rgb[1] < 0.1f);
    const char *tab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int val = 0, valb = -6;
    std::string enc;
    for (uint8_t c : raw) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            enc.push_back(tab[(val >> valb) & 63]);
            valb -= 6;
        }
    }
    if (valb > -6)
        enc.push_back(tab[((val << 8) >> (valb + 8)) & 63]);
    while (enc.size() % 4)
        enc.push_back('=');
    std::vector<float> rgb2;
    CHECK(decode_image_url(std::string("data:image/ppm;base64,") + enc, rgb2, iw, ih, ierr) ==
          Status::Ok);
    CHECK(iw == 2 && ih == 2);
    CHECK(decode_image_url("https://example.com/x.png", rgb2, iw, ih, ierr) == Status::Unsupported);

    static const uint8_t png_2x2[] = {
        0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a,0x00,0x00,0x00,0x0d,0x49,0x48,0x44,0x52,
        0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x08,0x02,0x00,0x00,0x00,0xfd,0xd4,0x9a,
        0x73,0x00,0x00,0x00,0x14,0x49,0x44,0x41,0x54,0x78,0xda,0x63,0xf8,0xcf,0xc0,0xc0,
        0x00,0xc2,0x0c,0xff,0xff,0xff,0x67,0x00,0x00,0x1e,0xef,0x04,0xfc,0x73,0x1c,0x53,
        0xcc,0x00,0x00,0x00,0x00,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82};
    std::vector<float> png_rgb;
    CHECK(decode_image_bytes(png_2x2, sizeof(png_2x2), png_rgb, iw, ih, ierr) == Status::Ok);
    CHECK(iw == 2 && ih == 2 && png_rgb.size() == 12);
    CHECK(png_rgb[0] > 0.9f && png_rgb[1] < 0.1f && png_rgb[2] < 0.1f);
    CHECK(png_rgb[3] < 0.1f && png_rgb[4] > 0.9f);

    static const uint8_t jpg_2x2[] = {
        0xff,0xd8,0xff,0xe0,0x00,0x10,0x4a,0x46,0x49,0x46,0x00,0x01,0x01,0x00,0x00,0x01,
        0x00,0x01,0x00,0x00,0xff,0xdb,0x00,0x43,0x00,0x02,0x01,0x01,0x01,0x01,0x01,0x02,
        0x01,0x01,0x01,0x02,0x02,0x02,0x02,0x02,0x04,0x03,0x02,0x02,0x02,0x02,0x05,0x04,
        0x04,0x03,0x04,0x06,0x05,0x06,0x06,0x06,0x05,0x06,0x06,0x06,0x07,0x09,0x08,0x06,
        0x07,0x09,0x07,0x06,0x06,0x08,0x0b,0x08,0x09,0x0a,0x0a,0x0a,0x0a,0x0a,0x06,0x08,
        0x0b,0x0c,0x0b,0x0a,0x0c,0x09,0x0a,0x0a,0x0a,0xff,0xdb,0x00,0x43,0x01,0x02,0x02,
        0x02,0x02,0x02,0x02,0x05,0x03,0x03,0x05,0x0a,0x07,0x06,0x07,0x0a,0x0a,0x0a,0x0a,
        0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,
        0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,
        0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0x0a,0xff,
        0xc0,0x00,0x11,0x08,0x00,0x02,0x00,0x02,0x03,0x01,0x22,0x00,0x02,0x11,0x01,0x03,
        0x11,0x01,0xff,0xc4,0x00,0x1f,0x00,0x00,0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0a,0x0b,0xff,0xc4,0x00,0xb5,0x10,0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,
        0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7d,0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,
        0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
        0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,
        0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,
        0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
        0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,
        0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,
        0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
        0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,
        0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,
        0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa,0xff,0xc4,0x00,0x1f,0x01,0x00,
        0x03,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,
        0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0xff,0xc4,0x00,0xb5,0x11,
        0x00,0x02,0x01,0x02,0x04,0x04,0x03,0x04,0x07,0x05,0x04,0x04,0x00,0x01,0x02,0x77,
        0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
        0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
        0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
        0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
        0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
        0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
        0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
        0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
        0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
        0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
        0xf9,0xfa,0xff,0xda,0x00,0x0c,0x03,0x01,0x00,0x02,0x11,0x03,0x11,0x00,0x3f,0x00,
        0xfd,0x07,0xfd,0x80,0xfe,0x15,0xfc,0x30,0xd7,0xbf,0x61,0x2f,0x82,0x9a,0xe6,0xb9,
        0xf0,0xe3,0x41,0xbd,0xbd,0xbd,0xf8,0x49,0xe1,0xb9,0xef,0x2f,0x2e,0xf4,0x78,0x24,
        0x96,0x79,0x5f,0x4b,0xb7,0x67,0x91,0xdd,0x94,0x96,0x66,0x62,0x49,0x62,0x49,0x24,
        0x92,0x68,0xa2,0x8a,0xff,0x00,0x1f,0x38,0xcf,0xfe,0x4b,0x0c,0xc7,0xfe,0xbf,0xd6,
        0xff,0x00,0xd3,0x92,0x3f,0x9a,0xf8,0x87,0xfe,0x47,0xf8,0xbf,0xfa,0xfb,0x53,0xff,
        0x00,0x4b,0x67,0xff,0xd9};
    std::vector<float> jpg_rgb;
    CHECK(decode_image_bytes(jpg_2x2, sizeof(jpg_2x2), jpg_rgb, iw, ih, ierr) == Status::Ok);
    CHECK(iw == 2 && ih == 2 && jpg_rgb.size() == 12);
    for (float v : jpg_rgb)
        CHECK(std::isfinite(v) && v >= 0.f && v <= 1.f);
}

static std::string gpt2_byte_token(int b) {
    int extra = 0;
    for (int i = 0; i < b; ++i) {
        bool d = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
        if (!d)
            ++extra;
    }
    bool db = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    uint32_t cp = db ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + extra);
    std::string s;
    if (cp < 0x80)
        s.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
    return s;
}

static void test_tokenizer() {
    using namespace mvllm;
    std::string err;
    Tokenizer fb;
    CHECK(fb.load("/tmp/does-not-exist-mvllm-tok", err) == Status::Ok);
    std::vector<int> ids;
    CHECK(fb.encode("Ab", ids) == Status::Ok);
    CHECK(ids.size() == 2 && ids[0] == 'A' && ids[1] == 'b');
    std::string back;
    CHECK(fb.decode(ids, back) == Status::Ok && back == "Ab");

    // Rank-BPE: merge pair whose concatenation has the lowest vocab id.
    std::string dir = tmpdir();
    std::string vocab;
    for (int b = 0; b < 256; ++b) {
        if (b)
            vocab += ",";
        vocab += "\"" + json_escape(gpt2_byte_token(b)) + "\":" + std::to_string(b);
    }
    // "ab" as a token with id 4 so it wins over other pair ids (>=5).
    vocab += ",\"" + json_escape(gpt2_byte_token('a') + gpt2_byte_token('b')) + "\":4";
    write_file(dir + "/tokenizer.json",
               std::string("{\"model\":{\"type\":\"BPE\",\"vocab\":{") + vocab +
                   "},\"merges\":[]},\"added_tokens\":[{\"id\":300,\"content\":\"<|user|>\"}],"
                   "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\","
                   "\"pattern\":{\"Regex\":\"\\\\p{Han}+\"}}]}}");
    Tokenizer tk;
    CHECK(tk.load(dir, err) == Status::Ok);
    CHECK(tk.rank_bpe());
    CHECK(tk.kimi());
    CHECK(tk.encode("ab", ids) == Status::Ok);
    CHECK(ids.size() == 1 && ids[0] == 4);
    CHECK(tk.decode(ids, back) == Status::Ok && back == "ab");
    CHECK(tk.encode("a<|user|>b", ids) == Status::Ok);
    CHECK(ids.size() == 3 && ids[0] == static_cast<int>('a') && ids[1] == 300 &&
          ids[2] == static_cast<int>('b'));

    std::string glmimg = tk.apply_chat(Family::Glm53, {{"user", "see <image> now"}}, false);
    CHECK(glmimg.find("<|begin_of_image|><image><|end_of_image|>") != std::string::npos);
    ChatMessage imsg;
    imsg.role = "user";
    imsg.content = "look";
    imsg.image_urls.push_back("data:image/ppm;base64,QQ==");
    std::string glmi = tk.apply_chat(Family::Glm53, {imsg}, false);
    CHECK(glmi.find("<|begin_of_image|><image><|end_of_image|>") != std::string::npos);
    std::string glm = tk.apply_chat(Family::Glm53, {{"user", "hi"}}, false);
    CHECK(glm.find("[gMASK]<sop>") == 0);
    CHECK(glm.find("<|user|>hi") != std::string::npos);
    CHECK(glm.find("<|assistant|><think></think>") != std::string::npos);
    CHECK(glm.find("Reasoning Effort") == std::string::npos);
    std::string glmt = tk.apply_chat(Family::Glm53, {{"user", "hi"}}, true);
    CHECK(glmt.find("<|system|>Reasoning Effort: Max") != std::string::npos);
    CHECK(glmt.find("<|assistant|><think>") != std::string::npos);
    CHECK(glmt.find("<think></think>") == std::string::npos);
    std::string glml = tk.apply_chat(Family::Glm53, {{"user", "hi"}}, true, "low");
    CHECK(glml.find("Reasoning Effort: Low") != std::string::npos);
    CHECK(glml.find("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>hi<|assistant|><think>") ==
          0);
    std::string k3 = tk.apply_chat(Family::KimiK3, {{"user", "hi"}}, false);
    CHECK(k3.find("<|im_start|>user") != std::string::npos);
    CHECK(k3.find("<|im_start|>assistant") != std::string::npos);

    // Official Kimi pretok: contraction stays on the word; leading punct joins.
    vocab += ",\"" + json_escape(std::string("'ll")) + "\":5";
    vocab += ",\"" + json_escape(std::string("\"H")) + "\":6";
    vocab += ",\"" + json_escape(std::string("ll")) + "\":7";
    write_file(dir + "/tokenizer.json",
               std::string("{\"model\":{\"type\":\"BPE\",\"vocab\":{") + vocab +
                   "},\"merges\":[]},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":"
                   "[{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\p{Han}+\"}}]}}");
    Tokenizer km;
    CHECK(km.load(dir, err) == Status::Ok);
    CHECK(km.kimi());
    CHECK(km.encode("I'll", ids) == Status::Ok);
    CHECK(ids.size() == 2 && ids[0] == static_cast<int>('I') && ids[1] == 5);
    CHECK(km.encode("\"Hello", ids) == Status::Ok);
    CHECK(!ids.empty() && ids[0] == 6);
}

static void test_k3_xtml() {
    using namespace mvllm;
    std::string dir = tmpdir();
    std::string vocab;
    for (int b = 0; b < 256; ++b) {
        if (b)
            vocab += ",";
        vocab += "\"" + json_escape(gpt2_byte_token(b)) + "\":" + std::to_string(b);
    }
    write_file(dir + "/tokenizer.json",
               std::string("{\"model\":{\"type\":\"BPE\",\"vocab\":{") + vocab +
                   "},\"merges\":[]},\"added_tokens\":["
                   "{\"id\":300,\"content\":\"<|open|>\"},"
                   "{\"id\":301,\"content\":\"<|close|>\"},"
                   "{\"id\":302,\"content\":\"<|sep|>\"},"
                   "{\"id\":303,\"content\":\"<|end_of_msg|>\"}"
                   "]}");
    Tokenizer tk;
    std::string err;
    CHECK(tk.load(dir, err) == Status::Ok);
    CHECK(tk.has_xtml());
    CHECK(tk.id_of("<|open|>") == 300 && tk.id_of("<|end_of_msg|>") == 303);

    std::string rendered = tk.apply_chat(Family::KimiK3, {{"user", "hi"}}, false);
    CHECK(rendered.find("<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>") !=
          std::string::npos);
    CHECK(rendered.find("<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>") !=
          std::string::npos);
    std::string thinkp = tk.apply_chat(Family::KimiK3, {{"user", "hi"}}, true);
    CHECK(thinkp.find("<|open|>think<|sep|>") != std::string::npos);

    std::vector<int> ids;
    CHECK(tk.encode_chat(Family::KimiK3, {{"user", "hi"}}, false, ids) == Status::Ok);
    CHECK(!ids.empty() && ids[0] == 300);
    int n_open = 0, n_sep = 0, n_close = 0, n_eom = 0;
    for (int id : ids) {
        n_open += id == 300;
        n_close += id == 301;
        n_sep += id == 302;
        n_eom += id == 303;
    }
    CHECK(n_open == 3); // user message, assistant message, response
    CHECK(n_close == 1);
    CHECK(n_eom == 1);
    CHECK(n_sep >= 4);

    // Segmented attr encode is the contract: " role" / "=\"" / role / "\"" are
    // separate encode() calls, not one string.
    std::vector<int> glued, a, b, c, d;
    tk.encode(" role=\"user\"", glued);
    tk.encode(" role", a);
    tk.encode("=\"", b);
    tk.encode("user", c);
    tk.encode("\"", d);
    std::vector<int> segs = a;
    segs.insert(segs.end(), b.begin(), b.end());
    segs.insert(segs.end(), c.begin(), c.end());
    segs.insert(segs.end(), d.begin(), d.end());
    bool found = false;
    for (size_t i = 0; i + segs.size() <= ids.size(); ++i) {
        if (std::equal(segs.begin(), segs.end(), ids.begin() + static_cast<std::ptrdiff_t>(i))) {
            found = true;
            break;
        }
    }
    CHECK(found);

    ChatMessage past;
    past.role = "assistant";
    past.content = "ok";
    past.reasoning = "hmm";
    std::vector<int> hist;
    CHECK(tk.encode_chat(Family::KimiK3, {{"user", "q"}, past}, false, hist) == Status::Ok);
    std::string hs;
    tk.decode(hist, hs);
    CHECK(hs.find("hmm") != std::string::npos);
    CHECK(hs.find("ok") != std::string::npos);

    ChatMessage tc;
    tc.role = "assistant";
    tc.tool_name = "get_weather";
    tc.content = "{\"q\":\"sf\"}";
    ChatMessage tr;
    tr.role = "tool";
    tr.tool_name = "get_weather";
    tr.content = "sunny";
    std::string tools = tk.apply_chat(Family::KimiK3, {{"user", "w?"}, tc, tr}, false);
    CHECK(tools.find("<|open|>tools<|sep|>") != std::string::npos);
    CHECK(tools.find("<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>") != std::string::npos);
    CHECK(tools.find("<|open|>json type=\"object\"<|sep|>") != std::string::npos);
    CHECK(tools.find("<|open|>message role=\"tool\" tool=\"get_weather\"") != std::string::npos);
    CHECK(tools.find("<|open|>tool_call") == std::string::npos);
    std::vector<int> tids;
    CHECK(tk.encode_chat(Family::KimiK3, {{"user", "w?"}, tc, tr}, false, tids) == Status::Ok);
    std::string td;
    tk.decode(tids, td);
    CHECK(td.find("get_weather") != std::string::npos);
    CHECK(td.find("sunny") != std::string::npos);

    K3ToolDecl decl;
    decl.name = "get_weather";
    decl.description = "Weather";
    decl.parameters_json = R"({"type":"object","properties":{"city":{"type":"string"}}})";
    std::vector<K3ToolDecl> decls{decl};
    std::string body = k3_tool_declare_body(decls);
    CHECK(body.find("# Tools") != std::string::npos);
    CHECK(body.find("get_weather") != std::string::npos);
    std::string with_decl = tk.apply_chat(Family::KimiK3, {{"user", "w?"}}, false, {}, &decls);
    CHECK(with_decl.find("type=\"tool-declare\"") != std::string::npos);
    CHECK(with_decl.find("# Tools") != std::string::npos);

    K3ToolCall call;
    call.name = "get_weather";
    call.index = 1;
    call.args.push_back({"city", "string", "Rome"});
    call.args.push_back({"days", "number", "1e2"});
    std::string blk = k3_render_tools_block({call});
    CHECK(blk.find("<|open|>tools<|sep|>") != std::string::npos);
    CHECK(blk.find("argument key=\"city\" type=\"string\"") != std::string::npos);
    std::string wire = "<|open|>tools<|sep|>"
                       "<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"
                       "<|open|>argument key=\"city\" type=\"string\"<|sep|>Rome<|close|>argument<|sep|>"
                       "<|close|>call<|sep|>"
                       "<|close|>tools<|sep|>";
    std::string content;
    std::vector<K3ParsedCall> parsed;
    CHECK(k3_parse_tool_calls("Sure." + wire, content, parsed));
    CHECK(parsed.size() == 1);
    CHECK(parsed[0].name == "get_weather");
    CHECK(parsed[0].arguments.find("Rome") != std::string::npos);
    CHECK(content.find("Sure") != std::string::npos);
    CHECK(content.find("<|open|>tools") == std::string::npos);

    std::vector<K3ToolDecl> fromj;
    CHECK(k3_extract_tools_json(
        R"({"tools":[{"type":"function","function":{"name":"get_weather","description":"W","parameters":{"type":"object"}}}]})",
        fromj));
    CHECK(fromj.size() == 1 && fromj[0].name == "get_weather");

    CHECK(k3_xtml_escape_attr("a&b\"c") == "a&amp;b&quot;c");
    CHECK(k3_xtml_unescape_attr("a&amp;b&quot;c") == "a&b\"c");
}

static void test_tiktoken_model() {
    using namespace mvllm;
    std::string dir = tmpdir();
    // raw bytes "h","i","hi" -> aA== / aQ== / aGk=
    write_file(dir + "/tiktoken.model", "aA== 0\naQ== 1\naGk= 2\n");
    write_file(dir + "/tokenizer_config.json", R"({
      "added_tokens_decoder": {
        "10": {"content": "<|open|>", "special": true},
        "11": {"content": "<|close|>", "special": true},
        "12": {"content": "<|sep|>", "special": true},
        "13": {"content": "<|end_of_msg|>", "special": true}
      }
    })");
    Tokenizer tk;
    std::string err;
    CHECK(tk.load(dir, err) == Status::Ok);
    CHECK(tk.from_tiktoken());
    CHECK(tk.rank_bpe());
    CHECK(tk.kimi());
    CHECK(tk.has_xtml());
    CHECK(tk.id_of("<|open|>") == 10);
    std::vector<int> ids;
    CHECK(tk.encode("hi", ids) == Status::Ok);
    CHECK(ids.size() == 1 && ids[0] == 2);
    std::string back;
    CHECK(tk.decode(ids, back) == Status::Ok && back == "hi");
    CHECK(tk.encode("h", ids) == Status::Ok);
    CHECK(ids.size() == 1 && ids[0] == 0);
    CHECK(tk.encode_chat(Family::KimiK3, {{"user", "hi"}}, false, ids) == Status::Ok);
    CHECK(!ids.empty() && ids[0] == 10);
}

static void test_glm_eos_ids() {
    using namespace mvllm;
    std::string dir = tmpdir();
    write_file(dir + "/config.json", R"({
      "model_type": "glm",
      "architectures": ["Glm5ForConditionalGeneration"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 40,
      "eos_token_id": [7, 8, 9]
    })");
    write_file(dir + "/generation_config.json", R"({"eos_token_id":[7,8,9,11]})");
    ModelConfig cfg;
    std::string err;
    CHECK(load_model_config(dir, cfg, err) == Status::Ok);
    CHECK(cfg.eos == 7);
    CHECK(cfg.eos_ids.size() == 4);
    CHECK(cfg.eos_ids[3] == 11);
    CHECK(is_stop_token(11, cfg, -1));
    CHECK(is_stop_token(4, cfg, 4));
    CHECK(!is_stop_token(3, cfg, -1));
}

static void test_config_and_families() {
    using namespace mvllm;
    std::string err;

    // K3 tiny
    std::string kdir = tmpdir();
    write_file(kdir + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 3,
      "vocab_size": 48,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_attention_heads": 4,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 8,
      "v_head_dim": 8,
      "num_experts": 4,
      "num_experts_per_token": 2,
      "moe_intermediate_size": 16,
      "routed_expert_hidden_size": 16,
      "num_shared_experts": 1,
      "attn_res_block_size": 2,
      "rms_norm_eps": 1e-5,
      "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "kda_layers": [1, 2],
        "full_attn_layers": [3],
        "use_full_rank_gate": true,
        "gate_lower_bound": -5.0
      },
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    CHECK(sniff_family(kdir) == Family::KimiK3);
    Engine ek;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    CHECK(ek.load(kdir, rt, err) == Status::Ok);
    GenParams gp;
    gp.max_new_tokens = 4;
    gp.eos = 1;
    GenResult gr;
    CHECK(ek.generate("hi", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
    CHECK(gr.completion_tokens <= 4);

    // GLM53 tiny
    std::string gdir = tmpdir();
    write_file(gdir + "/config.json", R"({
      "model_type": "glm",
      "architectures": ["Glm5ForConditionalGeneration"],
      "hidden_size": 32,
      "num_hidden_layers": 4,
      "vocab_size": 40,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_experts": 4,
      "num_experts_per_token": 2,
      "moe_intermediate_size": 16,
      "num_shared_experts": 1,
      "routed_scaling_factor": 2.5,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 16,
      "hc_mult": 2,
      "layer_types": ["linear","linear","linear","full_attention"],
      "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4},
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    CHECK(sniff_family(gdir) == Family::Glm53);
    Engine eg;
    CHECK(eg.load(gdir, rt, err) == Status::Ok);
    CHECK(eg.generate("ok", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);

    // H3
    std::string hdir = tmpdir();
    write_file(hdir + "/config.json", R"({"model_type":"minimax_h3","architectures":["MiniMaxH3"]})");
    CHECK(sniff_family(hdir) == Family::H3);
    Engine eh;
    CHECK(eh.load(hdir, rt, err) == Status::Ok);
    H3GenParams hp;
    hp.prompt = "fox";
    hp.steps = 2;
    hp.dit_layers = 4;
    hp.width = 32;
    hp.height = 32;
    hp.frames = 5;
    hp.output_path = hdir + "/out.txt";
    H3GenResult hr;
    CHECK(eh.generate_video(hp, hr, err) == Status::Ok);
    {
        TurnPerf hpf;
        eh.turn_perf(hpf, false);
        CHECK(hpf.t_attn > 0.0 || hpf.t_emm > 0.0);
        eh.turn_perf(hpf, true);
        eh.turn_perf(hpf, false);
        CHECK(hpf.t_attn == 0.0 && hpf.t_emm == 0.0);
    }
    CHECK(hr.blocks_streamed == 8); // 2 evals * 4 layers
    CHECK(!hr.output_path.empty());
    CHECK(hr.audio_used);
    CHECK(hr.note.find("audio=") != std::string::npos);
    CHECK(hr.note.find("pack=text+audio+video") != std::string::npos);

    // DSV4 tiny (official-shaped config, synthetic weights)
    std::string ddir = tmpdir();
    write_file(ddir + "/config.json", R"({
      "model_type": "deepseek_v4",
      "architectures": ["DeepseekV4ForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 40,
      "num_attention_heads": 2,
      "head_dim": 16,
      "q_lora_rank": 16,
      "qk_rope_head_dim": 8,
      "o_groups": 1,
      "o_lora_rank": 16,
      "sliding_window": 8,
      "index_n_heads": 2,
      "index_head_dim": 16,
      "index_topk": 2,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "n_shared_experts": 1,
      "moe_intermediate_size": 16,
      "hc_mult": 2,
      "hc_sinkhorn_iters": 3,
      "rms_norm_eps": 1e-6,
      "routed_scaling_factor": 1.5,
      "swiglu_limit": 10.0,
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    CHECK(sniff_family(ddir) == Family::Dsv4);
    Engine ed;
    CHECK(ed.load(ddir, rt, err) == Status::Ok);
    CHECK(ed.generate("ok", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
}

static void test_dsv4_tiny() {
    using namespace mvllm;
    std::string err;
    CHECK(std::string(family_name(Family::Dsv4)) == "dsv4");
    CHECK(sniff_family("/tmp/does-not-exist-dsv4-flash") == Family::Dsv4);
    CHECK(sniff_family("/tmp/does-not-exist-deepseek-v4") == Family::Dsv4);
    CHECK(sniff_family("/tmp/does-not-exist-deepseek_v4") == Family::Dsv4);

    std::string llama_dir = tmpdir();
    write_file(llama_dir + "/config.json",
               R"({"model_type":"llama","architectures":["LlamaForCausalLM"]})");
    CHECK(sniff_family(llama_dir) == Family::Llama);

    std::string glm_dir = tmpdir();
    write_file(glm_dir + "/config.json",
               R"({"model_type":"glm","architectures":["Glm5ForConditionalGeneration"]})");
    CHECK(sniff_family(glm_dir) == Family::Glm53);

    std::string kimi_dir = tmpdir();
    write_file(kimi_dir + "/config.json",
               R"({"model_type":"kimi_linear","architectures":["KimiLinearForCausalLM"]})");
    CHECK(sniff_family(kimi_dir) == Family::KimiK3);

    std::string full = tmpdir();
    write_file(full + "/config.json", R"({
      "model_type": "deepseek_v4",
      "architectures": ["DeepSeekV4ForCausalLM"],
      "hidden_size": 4096
    })");
    CHECK(sniff_family(full) == Family::Dsv4);
    ModelConfig fcfg;
    CHECK(load_model_config(full, fcfg, err) == Status::Ok);
    CHECK(fcfg.family == Family::Dsv4);
    CHECK(fcfg.hidden == 4096);
    CHECK(fcfg.n_layers == 43);
    CHECK(fcfg.moe.n_experts == 256);
    CHECK(fcfg.moe.topk == 6);
    CHECK(fcfg.moe.n_shared == 1);
    CHECK(fcfg.mla.q_lora == 1024);
    CHECK(fcfg.o_lora == 1024);
    CHECK(fcfg.sliding_window == 128);
    CHECK(fcfg.dsa.topk == 512);

    std::string tiny = tmpdir();
    write_file(tiny + "/config.json", R"({
      "architectures": ["DeepseekV4ForCausalLM"],
      "model_type": "deepseek_v4",
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "num_attention_heads": 2,
      "head_dim": 16,
      "q_lora_rank": 16,
      "qk_rope_head_dim": 8,
      "o_groups": 1,
      "o_lora_rank": 16,
      "sliding_window": 8,
      "index_n_heads": 2,
      "index_head_dim": 16,
      "index_topk": 2,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "n_shared_experts": 1,
      "moe_intermediate_size": 16,
      "hc_mult": 2,
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    ModelConfig tcfg;
    CHECK(load_model_config(tiny, tcfg, err) == Status::Ok);
    CHECK(tcfg.family == Family::Dsv4);
    CHECK(tcfg.hidden == 32);
    CHECK(tcfg.n_layers == 2);
    CHECK(tcfg.moe.n_experts == 4);
    CHECK(tcfg.moe.topk == 2);
    CHECK(tcfg.moe.n_shared == 1);
    CHECK(tcfg.head_dim == 16);
    CHECK(tcfg.mla.kv_lora == 16);
    CHECK(tcfg.o_lora == 16);
    CHECK(tcfg.sliding_window == 8);
    CHECK(tcfg.dsa.topk == 2);
    CHECK(tcfg.dsa.n_heads == 2);
    CHECK(tcfg.dsa.head_dim == 16);

    std::string official = tmpdir();
    write_file(official + "/config.json", R"({
      "model_type": "deepseek_v4",
      "architectures": ["DeepseekV4ForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 1,
      "vocab_size": 16,
      "n_routed_experts": 3,
      "num_experts_per_tok": 2,
      "n_shared_experts": 1,
      "index_topk": 4,
      "index_n_heads": 2,
      "index_head_dim": 8,
      "q_lora_rank": 8,
      "kv_lora": 12,
      "qk_rope_head_dim": 4,
      "o_lora_rank": 8,
      "o_groups": 2,
      "sliding_window": 7,
      "moe_intermediate_size": 16,
      "routed_scaling_factor": 1.25,
      "swiglu_limit": 7.0,
      "hc_mult": 3,
      "hc_sinkhorn_iters": 5,
      "hc_eps": 1e-4,
      "rope_theta": 50000,
      "rms_norm_eps": 1e-6,
      "original_max_position_embeddings": 4096
    })");
    CHECK(sniff_family(official) == Family::Dsv4);
    ModelConfig ocfg;
    CHECK(load_model_config(official, ocfg, err) == Status::Ok);
    CHECK(ocfg.family == Family::Dsv4);
    CHECK(ocfg.moe.n_experts == 3);
    CHECK(ocfg.moe.topk == 2);
    CHECK(ocfg.moe.n_shared == 1);
    CHECK(ocfg.dsa.topk == 4);
    CHECK(ocfg.dsa.n_heads == 2);
    CHECK(ocfg.dsa.head_dim == 8);
    CHECK(ocfg.mla.q_lora == 8);
    CHECK(ocfg.mla.kv_lora == 12);
    CHECK(ocfg.mla.qk_rope == 4);
    CHECK(ocfg.o_lora == 8);
    CHECK(ocfg.o_groups == 2);
    CHECK(ocfg.sliding_window == 7);
    CHECK(ocfg.moe.intermediate == 16);
    CHECK(ocfg.moe.routed_scale > 1.2f && ocfg.moe.routed_scale < 1.3f);
    CHECK(ocfg.moe.swiglu_limit == 7.f);
    CHECK(ocfg.mhc.mult == 3);
    CHECK(ocfg.mhc.iters == 5);
    CHECK(ocfg.max_position == 4096);

    Engine e;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    CHECK(e.load(tiny, rt, err) == Status::Ok);
    std::string info = e.info();
    CHECK(info.find("dsv4") != std::string::npos);
    CHECK(info.find("checkpoint=synthetic") != std::string::npos);
    const bool dsv4_cu_wire_tier_cpu = info.find("tier=cpu") != std::string::npos;
    CHECK(dsv4_cu_wire_tier_cpu);
    const bool dsv4_cu_wire_avail = mvllm::dsv4_cuda::available();
    CHECK(dsv4_cu_wire_avail);
    const bool dsv4_cu_wire_name_cpu =
        std::strcmp(mvllm::dsv4_cuda::backend_name(), "cpu") == 0;
    CHECK(dsv4_cu_wire_name_cpu);
    GenParams gp;
    gp.max_new_tokens = 4;
    gp.eos = 1;
    gp.cache_slot = 0;
    GenResult gr;
    CHECK(e.generate("hi", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
    CHECK(gr.completion_tokens <= 4);
    CHECK(gr.prompt_tokens > 0);
    const char *dsv4_draft_old = std::getenv("V4_DRAFT");
    const char *dsv4_chunk_old = std::getenv("COLI_PREFILL_CHUNK");
    const std::string dsv4_draft_prev = dsv4_draft_old ? dsv4_draft_old : "";
    const std::string dsv4_chunk_prev = dsv4_chunk_old ? dsv4_chunk_old : "";
    setenv("V4_DRAFT", "2", 1);
    setenv("COLI_PREFILL_CHUNK", "2", 1);
    GenResult gr_d;
    const bool dsv4_draft_ok = e.generate("hi", gp, gr_d, err) == Status::Ok;
    CHECK(dsv4_draft_ok);
    CHECK(gr_d.completion_tokens > 0);
    const std::string dsv4_draft_info = e.info();
    CHECK(dsv4_draft_info.find("draft=2") != std::string::npos);
    if (!dsv4_draft_prev.empty())
        setenv("V4_DRAFT", dsv4_draft_prev.c_str(), 1);
    else
        unsetenv("V4_DRAFT");
    if (!dsv4_chunk_prev.empty())
        setenv("COLI_PREFILL_CHUNK", dsv4_chunk_prev.c_str(), 1);
    else
        unsetenv("COLI_PREFILL_CHUNK");
    const char *dsv4_mtp_old = std::getenv("V4_MTP");
    const std::string dsv4_mtp_prev = dsv4_mtp_old ? dsv4_mtp_old : "";
    setenv("V4_MTP", "1", 1);
    unsetenv("V4_DRAFT");
    GenResult gr_m;
    const bool dsv4_mtp_ok = e.generate("hi", gp, gr_m, err) == Status::Ok;
    CHECK(dsv4_mtp_ok);
    CHECK(gr_m.completion_tokens > 0);
    const std::string dsv4_mtp_info = e.info();
    CHECK(dsv4_mtp_info.find("draft=3") != std::string::npos);
    CHECK(dsv4_mtp_info.find("mtp=") != std::string::npos);
    CHECK(dsv4_mtp_info.find("vk=") != std::string::npos);
    if (!dsv4_mtp_prev.empty())
        setenv("V4_MTP", dsv4_mtp_prev.c_str(), 1);
    else
        unsetenv("V4_MTP");

    FamilyEngine *fe = e.family_impl();
    CHECK(fe != nullptr);
    GenParams stream;
    stream.max_new_tokens = 3;
    stream.eos = 1;
    int reuse = 0;
    const std::vector<int> ids = {1, 2, 3};
    CHECK(fe->begin_generate(0, ids, stream, reuse, err) == Status::Ok);
    int nstep = 0;
    for (;;) {
        int tok = -1;
        bool done = false;
        CHECK(fe->next_token(0, tok, done, err) == Status::Ok);
        ++nstep;
        if (done)
            break;
        CHECK(nstep < 8);
    }
    CHECK(nstep > 0);
    fe->end_generate(0);

    KvPersistRecord row;
    const int nL = std::max(e.config().n_layers, 1);
    const int kvL = std::max(e.config().mla.kv_lora, 0);
    const int qr = std::max(e.config().mla.qk_rope, 0);
    const int idh = std::max(e.config().dsa.head_dim, 0);
    row.L.assign(static_cast<size_t>(nL) * static_cast<size_t>(std::max(kvL, 1)), 0.f);
    row.R.assign(static_cast<size_t>(nL) * static_cast<size_t>(std::max(qr, 1)), 0.f);
    row.I.assign(static_cast<size_t>(nL) * static_cast<size_t>(std::max(idh, 1)), 0.f);
    CHECK(fe->export_kv_rows(0, 0, 1, &row) > 0);
    CHECK(fe->export_kv_rows(-1, 0, 1, &row) == 0);
    CHECK(fe->export_kv_rows(99, 0, 1, &row) == 0);
    CHECK(fe->export_kv_rows(0, 0, 1, nullptr) == 0);
    CHECK(fe->import_kv_rows(1, 0, 1, &row) == 1);
    KvPersistRecord back;
    back.L.assign(row.L.size(), 0.f);
    back.R.assign(row.R.size(), 0.f);
    back.I.assign(row.I.size(), 0.f);
    CHECK(fe->export_kv_rows(1, 0, 1, &back) == 1);
    CHECK(back.L == row.L);
    CHECK(back.R == row.R);
    CHECK(back.I == row.I);
    CHECK(fe->import_kv_rows(-1, 0, 1, &row) == 0);

    int reuse0 = 0, reuse1 = 0;
    CHECK(fe->begin_generate(0, ids, stream, reuse0, err) == Status::Ok);
    CHECK(fe->begin_generate(1, {2, 3}, stream, reuse1, err) == Status::Ok);
    int mux_slots[2] = {0, 1};
    int mux_toks[2] = {-1, -1};
    uint8_t mux_done[2] = {0, 0};
    CHECK(fe->next_tokens(mux_slots, 2, mux_toks, mux_done, err) == Status::Ok);
    fe->end_generate(0);
    fe->end_generate(1);

    Tokenizer tk;
    std::string chat = tk.apply_chat(Family::Dsv4, {{"user", "hi"}}, false);
    CHECK(chat.find("<｜User｜>") != std::string::npos);
    CHECK(chat.find("<｜Assistant｜>") != std::string::npos);
    std::string think = tk.apply_chat(Family::Dsv4, {{"user", "hi"}}, true);
    CHECK(think.find("<think>") != std::string::npos);
}

static std::string env_copy(const char *k) {
    const char *v = std::getenv(k);
    return v ? std::string(v) : std::string();
}

static void env_restore(const char *k, const std::string &prev, bool had) {
    if (had)
        setenv(k, prev.c_str(), 1);
    else
        unsetenv(k);
}

static bool parse_dsv4_ckpt(const std::string &d, int &ckpt, unsigned long long &hits) {
    const auto cpos = d.find("ckpt=");
    const auto hpos = d.find("hits=");
    if (cpos == std::string::npos || hpos == std::string::npos)
        return false;
    ckpt = std::atoi(d.c_str() + cpos + 5);
    hits = std::strtoull(d.c_str() + hpos + 5, nullptr, 10);
    return true;
}

static bool dir_has_files(const std::string &p) {
    DIR *d = ::opendir(p.c_str());
    if (!d)
        return false;
    bool any = false;
    while (dirent *e = ::readdir(d)) {
        if (e->d_name[0] == '.')
            continue;
        any = true;
        break;
    }
    ::closedir(d);
    return any;
}

static void test_dsv4_prefix_ckpt() {
    using namespace mvllm;
    const char *keys[] = {"V4_PREFIX_CKPT",         "V4_PREFIX_CKPT_MIN", "V4_PREFIX_CKPT_DISK",
                          "V4_PREFIX_CKPT_SLOTS",   "V4_PREFIX_LOG",     "MVLLM_PREFIX_CKPT",
                          "MVLLM_PREFIX_CKPT_MIN",  "MVLLM_PREFIX_CKPT_DISK",
                          "MVLLM_PREFIX_CKPT_SLOTS", "MVLLM_PREFIX_CKPT_LOG"};
    std::string prev[10];
    bool had[10];
    for (int i = 0; i < 10; ++i) {
        had[i] = std::getenv(keys[i]) != nullptr;
        prev[i] = env_copy(keys[i]);
    }
    unsetenv("MVLLM_PREFIX_CKPT");
    unsetenv("MVLLM_PREFIX_CKPT_MIN");
    unsetenv("MVLLM_PREFIX_CKPT_DISK");
    unsetenv("MVLLM_PREFIX_CKPT_SLOTS");
    unsetenv("MVLLM_PREFIX_CKPT_LOG");
    setenv("V4_PREFIX_CKPT", "1", 1);
    setenv("V4_PREFIX_CKPT_MIN", "2", 1);
    setenv("V4_PREFIX_CKPT_DISK", "2", 1);
    setenv("V4_PREFIX_CKPT_SLOTS", "4", 1);

    std::string err;
    std::string tiny = tmpdir();
    write_file(tiny + "/config.json", R"({
      "architectures": ["DeepseekV4ForCausalLM"],
      "model_type": "deepseek_v4",
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "num_attention_heads": 2,
      "head_dim": 16,
      "q_lora_rank": 16,
      "qk_rope_head_dim": 8,
      "o_groups": 1,
      "o_lora_rank": 16,
      "sliding_window": 8,
      "index_n_heads": 2,
      "index_head_dim": 16,
      "index_topk": 2,
      "n_routed_experts": 4,
      "num_experts_per_tok": 2,
      "n_shared_experts": 1,
      "moe_intermediate_size": 16,
      "hc_mult": 2,
      "bos_token_id": 0,
      "eos_token_id": 1
    })");

    Engine e;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    CHECK(e.load(tiny, rt, err) == Status::Ok);
    FamilyEngine *fe = e.family_impl();
    CHECK(fe != nullptr);

    GenParams gp;
    gp.max_new_tokens = 1;
    gp.eos = 1;
    gp.prefix_reuse = 3;
    GenResult gr;
    const bool dsv4_ckpt_first_ok = fe->generate({2, 3, 4, 5}, gp, gr, err) == Status::Ok;
    CHECK(dsv4_ckpt_first_ok);
    int ckpt_n = 0;
    unsigned long long hits_n = 0;
    const bool dsv4_ckpt_desc1 = parse_dsv4_ckpt(fe->describe(), ckpt_n, hits_n);
    CHECK(dsv4_ckpt_desc1);
    const bool dsv4_ckpt_prompt_end = ckpt_n >= 1;
    CHECK(dsv4_ckpt_prompt_end);

    gp.prefix_reuse = 0;
    const bool dsv4_ckpt_second_ok = fe->generate({2, 3, 4, 6}, gp, gr, err) == Status::Ok;
    CHECK(dsv4_ckpt_second_ok);
    const bool dsv4_ckpt_desc2 = parse_dsv4_ckpt(fe->describe(), ckpt_n, hits_n);
    CHECK(dsv4_ckpt_desc2);
    const bool dsv4_ckpt_hit_prefix3 = hits_n >= 1;
    CHECK(dsv4_ckpt_hit_prefix3);

    const bool dsv4_ckpt_disk_dir = dir_has_files(tiny + "/.coli_ckpt");
    CHECK(dsv4_ckpt_disk_dir);

    Engine e2;
    const bool dsv4_ckpt_reload_ok = e2.load(tiny, rt, err) == Status::Ok;
    CHECK(dsv4_ckpt_reload_ok);
    FamilyEngine *fe2 = e2.family_impl();
    CHECK(fe2 != nullptr);
    const bool dsv4_ckpt_third_ok = fe2->generate({2, 3, 4, 7}, gp, gr, err) == Status::Ok;
    CHECK(dsv4_ckpt_third_ok);
    int ckpt2 = 0;
    unsigned long long hits2 = 0;
    const bool dsv4_ckpt_desc3 = parse_dsv4_ckpt(fe2->describe(), ckpt2, hits2);
    CHECK(dsv4_ckpt_desc3);
    const bool dsv4_ckpt_lazy_hit = hits2 >= 1;
    CHECK(dsv4_ckpt_lazy_hit);

    setenv("V4_PREFIX_CKPT", "0", 1);
    const unsigned long long hits_before_off = hits2;
    const bool dsv4_ckpt_off_ok = fe2->generate({2, 3, 4, 8}, gp, gr, err) == Status::Ok;
    CHECK(dsv4_ckpt_off_ok);
    int ckpt_off = 0;
    unsigned long long hits_off = 0;
    const bool dsv4_ckpt_desc_off = parse_dsv4_ckpt(fe2->describe(), ckpt_off, hits_off);
    CHECK(dsv4_ckpt_desc_off);
    const bool dsv4_ckpt_disabled_no_new_hits = hits_off == hits_before_off;
    CHECK(dsv4_ckpt_disabled_no_new_hits);

    for (int i = 0; i < 10; ++i)
        env_restore(keys[i], prev[i], had[i]);
}

static void test_dsv4_cuda_tier() {
    using namespace mvllm;
    using namespace mvllm::dsv4_cuda;

    auto sqrt_softplus = [](float v) {
        float sp;
        if (v > 20.f)
            sp = v;
        else if (v < -20.f)
            sp = std::exp(v);
        else
            sp = std::log1p(std::exp(v));
        return std::sqrt(std::max(sp, 0.f));
    };
    auto all_finite = [](const float *p, int n) {
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(p[i]))
                return false;
        return true;
    };

    const bool dsv4_cu_init = init(nullptr, 0);
    CHECK(dsv4_cu_init);
    const bool dsv4_cu_avail = available();
    CHECK(dsv4_cu_avail);
    const bool dsv4_cu_name_cpu = std::strcmp(backend_name(), "cpu") == 0;
    CHECK(dsv4_cu_name_cpu);
    const bool dsv4_cu_arch = backend_arch_ok(0);
    CHECK(dsv4_cu_arch);
    const bool dsv4_cu_init_again = init(nullptr, 0);
    CHECK(dsv4_cu_init_again);

    {
        const float W[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                             0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const float x[4] = {1.f, 2.f, 3.f, 4.f};
        float y[4] = {};
        Tensor *tw = nullptr;
        const bool dsv4_cu_up_f32 = upload_f32(&tw, W, 4, 4, 0);
        CHECK(dsv4_cu_up_f32);
        const bool dsv4_cu_mv_f32 = matvec(tw, y, x);
        CHECK(dsv4_cu_mv_f32);
        CHECK_NEAR(y[0], 1.f, 1e-5);
        CHECK_NEAR(y[1], 2.f, 1e-5);
        CHECK_NEAR(y[2], 3.f, 1e-5);
        CHECK_NEAR(y[3], 4.f, 1e-5);
        tensor_free(tw);
    }

    {
        const int O = 8, I = 8;
        std::vector<uint8_t> w(static_cast<size_t>(O) * I);
        std::vector<uint8_t> sc(static_cast<size_t>(O) * ((I + 127) / 128), 127);
        std::vector<float> x(I), y(O), ref(O, 0.f);
        for (int o = 0; o < O; ++o)
            for (int i = 0; i < I; ++i)
                w[static_cast<size_t>(o) * I + i] =
                    static_cast<uint8_t>(((o * 17 + i * 13) % 120) + 1);
        for (int i = 0; i < I; ++i)
            x[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        const float tile = e8m0_decode(127);
        for (int o = 0; o < O; ++o)
            for (int i = 0; i < I; ++i)
                ref[static_cast<size_t>(o)] +=
                    x[static_cast<size_t>(i)] *
                    e4m3fn_decode(w[static_cast<size_t>(o) * I + i]) * tile;
        Tensor *tw = nullptr;
        const bool dsv4_cu_up_fp8 = upload_fp8(&tw, w.data(), sc.data(), O, I, 0);
        CHECK(dsv4_cu_up_fp8);
        const bool dsv4_cu_mv_fp8 = matvec(tw, y.data(), x.data());
        CHECK(dsv4_cu_mv_fp8);
        for (int o = 0; o < O; ++o)
            CHECK_NEAR(y[static_cast<size_t>(o)], ref[static_cast<size_t>(o)], 1e-4);
        tensor_free(tw);
    }

    {
        const int O = 32, I = 32;
        std::vector<uint8_t> packed(static_cast<size_t>(O) * (I / 2));
        std::vector<uint8_t> sc(static_cast<size_t>(O) * ((I + 31) / 32), 127);
        std::vector<float> x(I), y(O), ref(O, 0.f);
        for (size_t i = 0; i < packed.size(); ++i)
            packed[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
        for (int i = 0; i < I; ++i)
            x[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        quant::matmul_mxfp4(ref.data(), x.data(), packed.data(), sc.data(), 1, I, O);
        Tensor *tw = nullptr;
        const bool dsv4_cu_up_fp4 = upload_fp4(&tw, packed.data(), sc.data(), O, I, 0);
        CHECK(dsv4_cu_up_fp4);
        const bool dsv4_cu_mv_fp4 = matvec(tw, y.data(), x.data());
        CHECK(dsv4_cu_mv_fp4);
        for (int o = 0; o < O; ++o)
            CHECK_NEAR(y[static_cast<size_t>(o)], ref[static_cast<size_t>(o)], 1e-4);
        tensor_free(tw);
    }

    {
        const uint16_t Wb[4] = {0x3f80, 0, 0, 0x3f80};
        const float xb[2] = {1.5f, -2.f};
        float yb[2] = {};
        Tensor *tw = nullptr;
        const bool dsv4_cu_up_bf16 = upload_bf16(&tw, Wb, 2, 2, 0);
        CHECK(dsv4_cu_up_bf16);
        const bool dsv4_cu_mv_bf16 = matvec(tw, yb, xb);
        CHECK(dsv4_cu_mv_bf16);
        CHECK_NEAR(yb[0], 1.5f, 1e-5);
        CHECK_NEAR(yb[1], -2.f, 1e-5);
        tensor_free(tw);
    }

    {
        float ax[8];
        for (int i = 0; i < 8; ++i)
            ax[i] = 0.1f * static_cast<float>(i + 1);
        Activation *a0 = activation_create(0, 8);
        const bool dsv4_cu_act_a0 = a0 != nullptr;
        CHECK(dsv4_cu_act_a0);
        const bool dsv4_cu_act_up = activation_upload(a0, ax, 8);
        CHECK(dsv4_cu_act_up);
        Activation *a1 = activation_create(0, 8);
        const bool dsv4_cu_act_a1 = a1 != nullptr;
        CHECK(dsv4_cu_act_a1);
        const bool dsv4_cu_act_cp = activation_copy(a1, a0, 8);
        CHECK(dsv4_cu_act_cp);
        float ay[8] = {};
        const bool dsv4_cu_act_dn = activation_download(ay, a1, 8);
        CHECK(dsv4_cu_act_dn);
        for (int i = 0; i < 8; ++i)
            CHECK_NEAR(ay[i], ax[i], 1e-5);
        const bool dsv4_cu_act_dev = activation_device(a0) == 0;
        CHECK(dsv4_cu_act_dev);
        const bool dsv4_cu_act_sync = activation_sync(a0);
        CHECK(dsv4_cu_act_sync);
        const bool dsv4_cu_act_n = activation_elements(a0) == 8;
        CHECK(dsv4_cu_act_n);

        Activation *a2 = activation_create(0, 8);
        const bool dsv4_cu_act_a2 = a2 != nullptr;
        CHECK(dsv4_cu_act_a2);
        float az[8] = {};
        const bool dsv4_cu_act_up2 = activation_upload(a2, az, 8);
        CHECK(dsv4_cu_act_up2);
        const bool dsv4_cu_act_rng = activation_copy_range(a2, 2, a0, 2, 3);
        CHECK(dsv4_cu_act_rng);
        float ar[8] = {};
        const bool dsv4_cu_act_dn2 = activation_download(ar, a2, 8);
        CHECK(dsv4_cu_act_dn2);
        CHECK_NEAR(ar[2], ax[2], 1e-5);
        CHECK_NEAR(ar[3], ax[3], 1e-5);
        CHECK_NEAR(ar[4], ax[4], 1e-5);

        const bool dsv4_cu_act_empty = activation_create(0, 0) == nullptr;
        CHECK(dsv4_cu_act_empty);
        const bool dsv4_cu_act_neg = activation_create(0, -3) == nullptr;
        CHECK(dsv4_cu_act_neg);
        activation_free(nullptr);
        activation_free(a0);
        activation_free(a1);
        activation_free(a2);
    }

    {
        const int O = 32, I = 32;
        const float limit = 7.f;
        const float ew[2] = {0.7f, 0.3f};
        std::vector<float> x(I);
        for (int i = 0; i < I; ++i)
            x[static_cast<size_t>(i)] = 0.05f * static_cast<float>((i % 7) - 3);
        Tensor *gate[2] = {}, *up[2] = {}, *down[2] = {};
        for (int e = 0; e < 2; ++e) {
            std::vector<uint8_t> pg(static_cast<size_t>(O) * (I / 2));
            std::vector<uint8_t> pu(static_cast<size_t>(O) * (I / 2));
            std::vector<uint8_t> pd(static_cast<size_t>(I) * (O / 2));
            std::vector<uint8_t> sg(static_cast<size_t>(O) * ((I + 31) / 32), 127);
            std::vector<uint8_t> su(static_cast<size_t>(O) * ((I + 31) / 32), 127);
            std::vector<uint8_t> sd(static_cast<size_t>(I) * ((O + 31) / 32), 127);
            for (size_t i = 0; i < pg.size(); ++i) {
                pg[i] = static_cast<uint8_t>((i * 19 + static_cast<size_t>(e) * 5 + 3) & 0xff);
                pu[i] = static_cast<uint8_t>((i * 23 + static_cast<size_t>(e) * 7 + 9) & 0xff);
            }
            for (size_t i = 0; i < pd.size(); ++i)
                pd[i] = static_cast<uint8_t>((i * 29 + static_cast<size_t>(e) * 11 + 1) & 0xff);
            const bool dsv4_cu_ex_g = upload_fp4(&gate[e], pg.data(), sg.data(), O, I, 0);
            CHECK(dsv4_cu_ex_g);
            const bool dsv4_cu_ex_u = upload_fp4(&up[e], pu.data(), su.data(), O, I, 0);
            CHECK(dsv4_cu_ex_u);
            const bool dsv4_cu_ex_d = upload_fp4(&down[e], pd.data(), sd.data(), I, O, 0);
            CHECK(dsv4_cu_ex_d);
        }

        std::vector<float> y_seq(I, 0.f);
        for (int e = 0; e < 2; ++e) {
            std::vector<float> g(O), u(O), h(O), d(I);
            const bool dsv4_cu_seq_g = matvec(gate[e], g.data(), x.data());
            CHECK(dsv4_cu_seq_g);
            const bool dsv4_cu_seq_u = matvec(up[e], u.data(), x.data());
            CHECK(dsv4_cu_seq_u);
            for (int i = 0; i < O; ++i)
                h[static_cast<size_t>(i)] =
                    quant::clamped_swiglu(g[static_cast<size_t>(i)], u[static_cast<size_t>(i)],
                                          limit);
            const bool dsv4_cu_seq_d = matvec(down[e], d.data(), h.data());
            CHECK(dsv4_cu_seq_d);
            for (int i = 0; i < I; ++i)
                y_seq[static_cast<size_t>(i)] += ew[e] * d[static_cast<size_t>(i)];
        }
        std::vector<float> y_grp(I, 0.f);
        const bool dsv4_cu_ex_grp = expert_group(gate, up, down, ew, 2, limit, y_grp.data(), x.data());
        CHECK(dsv4_cu_ex_grp);
        for (int i = 0; i < I; ++i)
            CHECK_NEAR(y_grp[static_cast<size_t>(i)], y_seq[static_cast<size_t>(i)], 1e-3);

        std::vector<uint8_t> sw(static_cast<size_t>(O) * I);
        std::vector<uint8_t> ssc(static_cast<size_t>(O) * ((I + 127) / 128), 127);
        for (size_t i = 0; i < sw.size(); ++i)
            sw[i] = static_cast<uint8_t>((i * 11 + 5) % 120 + 1);
        Tensor *sg = nullptr, *su = nullptr, *sd = nullptr;
        const bool dsv4_cu_sh_g = upload_fp8(&sg, sw.data(), ssc.data(), O, I, 0);
        CHECK(dsv4_cu_sh_g);
        const bool dsv4_cu_sh_u = upload_fp8(&su, sw.data(), ssc.data(), O, I, 0);
        CHECK(dsv4_cu_sh_u);
        const bool dsv4_cu_sh_d = upload_fp8(&sd, sw.data(), ssc.data(), I, O, 0);
        CHECK(dsv4_cu_sh_d);
        std::vector<float> y_sh(I, 0.f), y_moe(I, 0.f);
        const bool dsv4_cu_ex_fp8 = expert_fp8(sg, su, sd, limit, y_sh.data(), x.data());
        CHECK(dsv4_cu_ex_fp8);
        const bool dsv4_cu_moe = moe(gate, up, down, ew, 2, sg, su, sd, limit, y_moe.data(), x.data());
        CHECK(dsv4_cu_moe);
        for (int i = 0; i < I; ++i)
            CHECK_NEAR(y_moe[static_cast<size_t>(i)],
                       y_grp[static_cast<size_t>(i)] + y_sh[static_cast<size_t>(i)], 1e-3);

        tensor_free(sg);
        tensor_free(su);
        tensor_free(sd);
        for (int e = 0; e < 2; ++e) {
            tensor_free(gate[e]);
            tensor_free(up[e]);
            tensor_free(down[e]);
        }
    }

    {
        const int E = 16, H = 8;
        const float routed_scale = 1.5f;
        std::vector<float> W(static_cast<size_t>(E) * H), bias(E), xv(H);
        for (int e = 0; e < E; ++e) {
            bias[static_cast<size_t>(e)] = static_cast<float>(e);
            for (int h = 0; h < H; ++h)
                W[static_cast<size_t>(e) * H + h] = 0.01f * static_cast<float>((e + h) % 3);
        }
        for (int h = 0; h < H; ++h)
            xv[static_cast<size_t>(h)] = 1.f;
        Tensor *tg = nullptr, *tb = nullptr;
        const bool dsv4_cu_rt_g = upload_f32(&tg, W.data(), E, H, 0);
        CHECK(dsv4_cu_rt_g);
        const bool dsv4_cu_rt_b = upload_f32(&tb, bias.data(), E, 1, 0);
        CHECK(dsv4_cu_rt_b);
        Activation *ain = activation_create(0, H);
        const bool dsv4_cu_rt_ain = ain != nullptr;
        CHECK(dsv4_cu_rt_ain);
        const bool dsv4_cu_rt_up = activation_upload(ain, xv.data(), H);
        CHECK(dsv4_cu_rt_up);

        float logits[16], scores[16], choice[16];
        quant::matmul_f32(logits, xv.data(), W.data(), 1, H, E);
        int order[16];
        for (int e = 0; e < E; ++e) {
            scores[e] = sqrt_softplus(logits[e]);
            choice[e] = scores[e] + bias[static_cast<size_t>(e)];
            order[e] = e;
        }
        std::partial_sort(order, order + 6, order + E, [&](int a, int b) {
            if (choice[a] == choice[b])
                return a < b;
            return choice[a] > choice[b];
        });
        int ref_ids[6];
        float ref_w[6];
        float wsum = 0.f;
        for (int k = 0; k < 6; ++k) {
            ref_ids[k] = order[k];
            ref_w[k] = scores[order[k]] < 0.f ? 0.f : scores[order[k]];
            wsum += ref_w[k];
        }
        for (int k = 0; k < 6; ++k)
            ref_w[k] = (wsum > 0.f ? ref_w[k] / wsum : 1.f / 6.f) * routed_scale;

        int ids[6] = {};
        float weights[6] = {};
        const bool dsv4_cu_route = route(ain, tg, tb, nullptr, routed_scale, ids, weights);
        CHECK(dsv4_cu_route);
        for (int k = 0; k < 6; ++k) {
            const bool dsv4_cu_rt_id = ids[k] == ref_ids[k];
            CHECK(dsv4_cu_rt_id);
            CHECK_NEAR(weights[k], ref_w[k], 1e-4);
        }

        const int fixed[6] = {1, 3, 5, 7, 9, 11};
        float fix_w[6];
        float fsum = 0.f;
        for (int k = 0; k < 6; ++k) {
            fix_w[k] = scores[fixed[k]] < 0.f ? 0.f : scores[fixed[k]];
            fsum += fix_w[k];
        }
        for (int k = 0; k < 6; ++k)
            fix_w[k] = (fsum > 0.f ? fix_w[k] / fsum : 1.f / 6.f) * routed_scale;
        int fids[6] = {};
        float fweights[6] = {};
        const bool dsv4_cu_route_fix = route(ain, tg, tb, fixed, routed_scale, fids, fweights);
        CHECK(dsv4_cu_route_fix);
        for (int k = 0; k < 6; ++k) {
            const bool dsv4_cu_rt_fid = fids[k] == fixed[k];
            CHECK(dsv4_cu_rt_fid);
            CHECK_NEAR(fweights[k], fix_w[k], 1e-4);
        }
        activation_free(ain);
        tensor_free(tg);
        tensor_free(tb);
    }

    {
        const int M = 2, H = 8;
        const int mix_n = (2 + M) * M;
        std::vector<float> residual(static_cast<size_t>(M) * H);
        std::vector<float> fn(static_cast<size_t>(mix_n) * M * H);
        std::vector<float> scale = {0.7f, 0.8f, 0.9f};
        std::vector<float> base(static_cast<size_t>(mix_n));
        for (int i = 0; i < M * H; ++i)
            residual[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        for (size_t i = 0; i < fn.size(); ++i)
            fn[i] = 0.01f * static_cast<float>(static_cast<int>(i % 5) - 2);
        for (int i = 0; i < mix_n; ++i)
            base[static_cast<size_t>(i)] = 0.02f * static_cast<float>(i - 3);
        Tensor *tfn = nullptr, *tsc = nullptr, *tbase = nullptr;
        const bool dsv4_cu_mhc_fn = upload_f32(&tfn, fn.data(), mix_n, M * H, 0);
        CHECK(dsv4_cu_mhc_fn);
        const bool dsv4_cu_mhc_sc = upload_f32(&tsc, scale.data(), 3, 1, 0);
        CHECK(dsv4_cu_mhc_sc);
        const bool dsv4_cu_mhc_base = upload_f32(&tbase, base.data(), mix_n, 1, 0);
        CHECK(dsv4_cu_mhc_base);
        Activation *ares = activation_create(0, static_cast<long long>(M) * H);
        Activation *astate = activation_create(0, M + M * M);
        Activation *ain = activation_create(0, H);
        const bool dsv4_cu_mhc_acts = ares && astate && ain;
        CHECK(dsv4_cu_mhc_acts);
        const bool dsv4_cu_mhc_up = activation_upload(ares, residual.data(), M * H);
        CHECK(dsv4_cu_mhc_up);
        const bool dsv4_cu_mhc_pre =
            mhc_pre(ares, tfn, tsc, tbase, M, H, 1e-6f, 1e-6f, 1e-6f, 1.f, 3, astate, ain);
        CHECK(dsv4_cu_mhc_pre);
        std::vector<float> collapsed(H, 0.f), stv(static_cast<size_t>(M + M * M), 0.f);
        const bool dsv4_cu_mhc_dn_in = activation_download(collapsed.data(), ain, H);
        CHECK(dsv4_cu_mhc_dn_in);
        const bool dsv4_cu_mhc_dn_st =
            activation_download(stv.data(), astate, M + M * M);
        CHECK(dsv4_cu_mhc_dn_st);
        bool any_nz = false;
        for (int i = 0; i < H; ++i)
            if (collapsed[static_cast<size_t>(i)] != 0.f)
                any_nz = true;
        const bool dsv4_cu_mhc_in_nz = any_nz;
        CHECK(dsv4_cu_mhc_in_nz);
        const bool dsv4_cu_mhc_in_fin = all_finite(collapsed.data(), H);
        CHECK(dsv4_cu_mhc_in_fin);
        const bool dsv4_cu_mhc_st_n = activation_elements(astate) == M + M * M;
        CHECK(dsv4_cu_mhc_st_n);
        const bool dsv4_cu_mhc_st_fin = all_finite(stv.data(), M + M * M);
        CHECK(dsv4_cu_mhc_st_fin);

        std::vector<float> branch(H);
        for (int i = 0; i < H; ++i)
            branch[static_cast<size_t>(i)] = 0.05f * static_cast<float>(i + 1);
        Activation *abr = activation_create(0, H);
        Activation *aout = activation_create(0, static_cast<long long>(M) * H);
        const bool dsv4_cu_mhc_br = abr && aout;
        CHECK(dsv4_cu_mhc_br);
        const bool dsv4_cu_mhc_br_up = activation_upload(abr, branch.data(), H);
        CHECK(dsv4_cu_mhc_br_up);
        const bool dsv4_cu_mhc_post = mhc_post(abr, ares, astate, M, H, aout);
        CHECK(dsv4_cu_mhc_post);
        std::vector<float> postv(static_cast<size_t>(M) * H, 0.f);
        const bool dsv4_cu_mhc_dn_out = activation_download(postv.data(), aout, M * H);
        CHECK(dsv4_cu_mhc_dn_out);
        const bool dsv4_cu_mhc_out_fin = all_finite(postv.data(), M * H);
        CHECK(dsv4_cu_mhc_out_fin);

        std::vector<float> zres(static_cast<size_t>(M) * H, 0.f);
        const bool dsv4_cu_mhc_zup = activation_upload(ares, zres.data(), M * H);
        CHECK(dsv4_cu_mhc_zup);
        const bool dsv4_cu_mhc_pre0 =
            mhc_pre(ares, tfn, tsc, tbase, M, H, 1e-6f, 1e-6f, 1e-6f, 1.f, 3, astate, ain);
        CHECK(dsv4_cu_mhc_pre0);
        std::fill(collapsed.begin(), collapsed.end(), 1.f);
        const bool dsv4_cu_mhc_dn0 = activation_download(collapsed.data(), ain, H);
        CHECK(dsv4_cu_mhc_dn0);
        for (int i = 0; i < H; ++i)
            CHECK_NEAR(collapsed[static_cast<size_t>(i)], 0.f, 1e-4);

        activation_free(ares);
        activation_free(astate);
        activation_free(ain);
        activation_free(abr);
        activation_free(aout);
        tensor_free(tfn);
        tensor_free(tsc);
        tensor_free(tbase);
    }

    {
        const int tokens = 2, heads = 2, dim = 4, value_rows = 3, comp_base = 2;
        std::vector<float> q(static_cast<size_t>(tokens) * heads * dim);
        std::vector<float> q2(q.size());
        std::vector<float> vals(static_cast<size_t>(value_rows) * dim);
        std::vector<float> sinks(heads);
        std::vector<float> out(static_cast<size_t>(tokens) * heads * dim, 0.f);
        std::vector<float> out2(out.size(), 0.f);
        for (size_t i = 0; i < q.size(); ++i) {
            q[i] = 0.1f * static_cast<float>(i + 1);
            q2[i] = 0.3f * static_cast<float>(i + 2);
        }
        for (size_t i = 0; i < vals.size(); ++i)
            vals[i] = 0.1f * static_cast<float>(i + 1);
        for (int h = 0; h < heads; ++h)
            sinks[static_cast<size_t>(h)] = 0.1f * static_cast<float>(h + 1);
        const int meta[6] = {0, 2, 1, 1, 1, 1};
        const float scale = 0.5f;
        const bool dsv4_cu_sa = sparse_attn_batch(0, q.data(), vals.data(), sinks.data(), meta,
                                                  value_rows, comp_base, heads, dim, tokens, scale,
                                                  out.data());
        CHECK(dsv4_cu_sa);
        const bool dsv4_cu_sa_fin = all_finite(out.data(), static_cast<int>(out.size()));
        CHECK(dsv4_cu_sa_fin);
        const bool dsv4_cu_sa2 = sparse_attn_batch(0, q2.data(), vals.data(), sinks.data(), meta,
                                                   value_rows, comp_base, heads, dim, tokens, scale,
                                                   out2.data());
        CHECK(dsv4_cu_sa2);
        bool changed = false;
        for (size_t i = 0; i < out.size(); ++i)
            if (std::fabs(out[i] - out2[i]) > 1e-6f)
                changed = true;
        const bool dsv4_cu_sa_changed = changed;
        CHECK(dsv4_cu_sa_changed);
    }

    {
        const int tokens = 1, heads = 2, dim = 4, count = 3;
        std::vector<float> queries(static_cast<size_t>(tokens) * heads * dim, 0.25f);
        std::vector<float> keys(static_cast<size_t>(count) * dim, 0.5f);
        std::vector<float> head_w(static_cast<size_t>(tokens) * heads, 1.f);
        const int counts[1] = {2};
        float scores[3] = {1.f, 1.f, 1.f};
        const bool dsv4_cu_idx = indexer_score_batch(0, queries.data(), keys.data(), head_w.data(),
                                                     counts, tokens, heads, dim, count, scores);
        CHECK(dsv4_cu_idx);
        CHECK_NEAR(scores[2], 0.f, 1e-6);
        const bool dsv4_cu_idx_pos = scores[0] > 0.f || scores[1] > 0.f;
        CHECK(dsv4_cu_idx_pos);

        std::vector<float> kneg(keys.size(), -0.5f);
        float sneg[3] = {1.f, 1.f, 1.f};
        const bool dsv4_cu_idx_relu =
            indexer_score_batch(0, queries.data(), kneg.data(), head_w.data(), counts, tokens, heads,
                                dim, count, sneg);
        CHECK(dsv4_cu_idx_relu);
        CHECK_NEAR(sneg[0], 0.f, 1e-6);
        CHECK_NEAR(sneg[1], 0.f, 1e-6);
        CHECK_NEAR(sneg[2], 0.f, 1e-6);
    }

    {
        const int heads = 2, dim = 4, tokens = 1;
        float rows[8];
        for (int i = 0; i < 8; ++i)
            rows[i] = 0.1f * static_cast<float>(i + 1);
        const bool dsv4_cu_kv_app = kv_ring_append(0, 0, rows, 0, 2, 8, dim);
        CHECK(dsv4_cu_kv_app);
        float q[8], chunk[8], sinks[2] = {0.1f, 0.2f}, out[8] = {};
        for (int i = 0; i < 8; ++i) {
            q[i] = 0.1f * static_cast<float>(i + 1);
            chunk[i] = rows[i];
        }
        const int meta[3] = {0, 2, 0};
        const bool dsv4_cu_sa_cached =
            sparse_attn_batch_cached(0, 0, q, chunk, 0, sinks, meta, 0, 0, heads, dim, tokens, 0.5f,
                                     out);
        CHECK(dsv4_cu_sa_cached);
        const bool dsv4_cu_sa_cached_fin = all_finite(out, 8);
        CHECK(dsv4_cu_sa_cached_fin);
    }

    {
        const bool dsv4_cu_graph_begin = graph_begin(0);
        CHECK(dsv4_cu_graph_begin);
        Graph *g = graph_end(0);
        const bool dsv4_cu_graph_end = g != nullptr;
        CHECK(dsv4_cu_graph_end);
        const bool dsv4_cu_graph_launch = graph_launch(g);
        CHECK(dsv4_cu_graph_launch);
        graph_free(g);
    }

    {
        const float W[16] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                             10.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float x[4] = {1.f, 0.f, 0.f, 0.f};
        Tensor *tw = nullptr;
        const bool dsv4_cu_ha_up = upload_f32(&tw, W, 4, 4, 0);
        CHECK(dsv4_cu_ha_up);
        int id = -1;
        float value = 0.f;
        const bool dsv4_cu_ha = head_argmax(tw, x, &id, &value);
        CHECK(dsv4_cu_ha);
        const bool dsv4_cu_ha_id = id == 2;
        CHECK(dsv4_cu_ha_id);
        tensor_free(tw);
    }

    {
        const int H = 8, heads = 2, dim = 4;
        std::vector<float> ones8(H, 1.f), ones4(dim, 1.f), ones2(heads, 0.f);
        std::vector<float> ident(static_cast<size_t>(H) * H, 0.f);
        for (int i = 0; i < H; ++i)
            ident[static_cast<size_t>(i) * H + i] = 1.f;
        std::vector<float> wkv(static_cast<size_t>(dim) * H, 0.f);
        for (int i = 0; i < dim; ++i)
            wkv[static_cast<size_t>(i) * H + i] = 1.f;
        std::vector<float> xin(H);
        for (int i = 0; i < H; ++i)
            xin[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        Tensor *attn_n = nullptr, *q_a = nullptr, *q_n = nullptr, *q_b = nullptr, *wkv_t = nullptr,
               *kv_n = nullptr, *sink = nullptr, *wo_a = nullptr, *wo_b = nullptr;
        const bool dsv4_cu_af_n = upload_f32(&attn_n, ones8.data(), 1, H, 0);
        CHECK(dsv4_cu_af_n);
        const bool dsv4_cu_af_qa = upload_f32(&q_a, ident.data(), H, H, 0);
        CHECK(dsv4_cu_af_qa);
        const bool dsv4_cu_af_qn = upload_f32(&q_n, ones8.data(), 1, H, 0);
        CHECK(dsv4_cu_af_qn);
        const bool dsv4_cu_af_qb = upload_f32(&q_b, ident.data(), heads * dim, H, 0);
        CHECK(dsv4_cu_af_qb);
        const bool dsv4_cu_af_wkv = upload_f32(&wkv_t, wkv.data(), dim, H, 0);
        CHECK(dsv4_cu_af_wkv);
        const bool dsv4_cu_af_kvn = upload_f32(&kv_n, ones4.data(), 1, dim, 0);
        CHECK(dsv4_cu_af_kvn);
        const bool dsv4_cu_af_sink = upload_f32(&sink, ones2.data(), heads, 1, 0);
        CHECK(dsv4_cu_af_sink);
        const bool dsv4_cu_af_woa = upload_f32(&wo_a, ident.data(), H, heads * dim, 0);
        CHECK(dsv4_cu_af_woa);
        const bool dsv4_cu_af_wob = upload_f32(&wo_b, ident.data(), H, H, 0);
        CHECK(dsv4_cu_af_wob);
        Activation *ain = activation_create(0, H);
        Activation *aout = activation_create(0, H);
        const bool dsv4_cu_af_acts = ain && aout;
        CHECK(dsv4_cu_af_acts);
        const bool dsv4_cu_af_up = activation_upload(ain, xin.data(), H);
        CHECK(dsv4_cu_af_up);
        const bool dsv4_cu_af =
            attention_first(ain, attn_n, q_a, q_n, q_b, wkv_t, kv_n, sink, wo_a, wo_b, heads, dim, 0,
                            1, 1e-6f, aout);
        CHECK(dsv4_cu_af);
        const bool dsv4_cu_af_len = activation_elements(aout) == H;
        CHECK(dsv4_cu_af_len);
        std::vector<float> y(H, 0.f);
        const bool dsv4_cu_af_dn = activation_download(y.data(), aout, H);
        CHECK(dsv4_cu_af_dn);
        const bool dsv4_cu_af_fin = all_finite(y.data(), H);
        CHECK(dsv4_cu_af_fin);
        activation_free(ain);
        activation_free(aout);
        tensor_free(attn_n);
        tensor_free(q_a);
        tensor_free(q_n);
        tensor_free(q_b);
        tensor_free(wkv_t);
        tensor_free(kv_n);
        tensor_free(sink);
        tensor_free(wo_a);
        tensor_free(wo_b);
    }

    tensor_free(nullptr);
    shutdown();
    (void)available();
    const bool dsv4_cu_reinit = init(nullptr, 0);
    CHECK(dsv4_cu_reinit);
    const bool dsv4_cu_reavail = available();
    CHECK(dsv4_cu_reavail);
    shutdown();
}

static void test_coli_cuda_tier() {
    using namespace mvllm;
    using namespace mvllm::coli_cuda;

    auto silu = [](float v) { return v * quant::sigmoid(v); };
    auto all_finite = [](const float *p, int n) {
        for (int i = 0; i < n; ++i)
            if (!std::isfinite(p[i]))
                return false;
        return true;
    };

    const bool coli_cu_w0 = weight_at_supported(0);
    CHECK(coli_cu_w0);
    const bool coli_cu_w1 = weight_at_supported(1);
    CHECK(coli_cu_w1);
    const bool coli_cu_w2 = weight_at_supported(2);
    CHECK(coli_cu_w2);
    const bool coli_cu_w3 = weight_at_supported(3);
    CHECK(coli_cu_w3);
    const bool coli_cu_w4 = weight_at_supported(4);
    CHECK(coli_cu_w4);
    const bool coli_cu_w5 = !weight_at_supported(5);
    CHECK(coli_cu_w5);
    const bool coli_cu_w7 = !weight_at_supported(7);
    CHECK(coli_cu_w7);
    const bool coli_cu_w6 = !weight_at_supported(6);
    CHECK(coli_cu_w6);
    const bool coli_cu_w8 = !weight_at_supported(8);
    CHECK(coli_cu_w8);

    const bool coli_cu_init = init(nullptr, 0);
    CHECK(coli_cu_init);
    const bool coli_cu_avail = available();
    CHECK(coli_cu_avail);
    const bool coli_cu_ndev = available_device_count() == 0;
    CHECK(coli_cu_ndev);
    const bool coli_cu_init2 = init(nullptr, 0);
    CHECK(coli_cu_init2);

    {
        const float W[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                             0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const float x[4] = {1.f, 2.f, 3.f, 4.f};
        float y[4] = {};
        Tensor *tw = nullptr;
        const bool coli_cu_up_f32 = tensor_upload(&tw, W, nullptr, 0, 4, 4, 0, 0);
        CHECK(coli_cu_up_f32);
        const bool coli_cu_mm_f32 = matmul(&tw, y, x, W, nullptr, 0, 1, 4, 4, 0, 0);
        CHECK(coli_cu_mm_f32);
        CHECK_NEAR(y[0], 1.f, 1e-5);
        CHECK_NEAR(y[1], 2.f, 1e-5);
        CHECK_NEAR(y[2], 3.f, 1e-5);
        CHECK_NEAR(y[3], 4.f, 1e-5);
        tensor_free(tw);
    }

    {
        const int O = 32, I = 32;
        std::vector<float> W(static_cast<size_t>(O) * I), x(I), y(O), ref(O);
        for (int i = 0; i < O * I; ++i)
            W[static_cast<size_t>(i)] = ((i * 17) % 11 - 5) * 0.1f;
        for (int i = 0; i < I; ++i)
            x[static_cast<size_t>(i)] = ((i * 3) % 7 - 3) * 0.2f;
        std::vector<uint8_t> packed(static_cast<size_t>(O) * (I / 2));
        std::vector<float> scales(static_cast<size_t>(O) * ((I + 63) / 64));
        quant::quantize_int4_g64(W.data(), O, I, packed.data(), scales.data());
        quant::matmul_int4_g64(ref.data(), x.data(), packed.data(), scales.data(), 1, I, O);
        Tensor *tw = nullptr;
        const bool coli_cu_up_i4 = tensor_upload(&tw, packed.data(), scales.data(), 4, I, O, 0, 64);
        CHECK(coli_cu_up_i4);
        const bool coli_cu_mm_i4 =
            matmul(&tw, y.data(), x.data(), packed.data(), scales.data(), 4, 1, I, O, 0, 64);
        CHECK(coli_cu_mm_i4);
        for (int o = 0; o < O; ++o)
            CHECK_NEAR(y[static_cast<size_t>(o)], ref[static_cast<size_t>(o)], 1e-4);
        tensor_free(tw);
    }

    {
        const int O = 32, I = 32;
        std::vector<uint8_t> packed(static_cast<size_t>(O) * (I / 2));
        std::vector<uint8_t> e8s(static_cast<size_t>(O) * ((I + 31) / 32), 127);
        std::vector<float> x(I), y(O), ref(O);
        for (size_t i = 0; i < packed.size(); ++i)
            packed[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
        for (int i = 0; i < I; ++i)
            x[static_cast<size_t>(i)] = 0.1f * static_cast<float>(i + 1);
        quant::matmul_mxfp4(ref.data(), x.data(), packed.data(), e8s.data(), 1, I, O);
        const bool coli_cu_mm_mx =
            matmul_mxfp4(y.data(), x.data(), packed.data(), e8s.data(), 1, I, O);
        CHECK(coli_cu_mm_mx);
        for (int o = 0; o < O; ++o)
            CHECK_NEAR(y[static_cast<size_t>(o)], ref[static_cast<size_t>(o)], 1e-4);
    }

    {
        const int D = 4;
        const float ident[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const float x[4] = {0.2f, -0.4f, 0.6f, -0.8f};
        float y[4] = {}, ref[4] = {};
        Tensor *gate = nullptr, *up = nullptr, *down = nullptr;
        const bool coli_cu_mlp_g = tensor_upload(&gate, ident, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_mlp_g);
        const bool coli_cu_mlp_u = tensor_upload(&up, ident, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_mlp_u);
        const bool coli_cu_mlp_d = tensor_upload(&down, ident, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_mlp_d);
        const bool coli_cu_mlp = expert_mlp(gate, up, down, y, x, 1);
        CHECK(coli_cu_mlp);
        for (int i = 0; i < D; ++i)
            ref[i] = silu(x[i]) * x[i];
        const bool coli_cu_mlp_fin = all_finite(y, D);
        CHECK(coli_cu_mlp_fin);
        for (int i = 0; i < D; ++i)
            CHECK_NEAR(y[i], ref[i], 1e-5);
        tensor_free(gate);
        tensor_free(up);
        tensor_free(down);
    }

    {
        const int D = 4;
        const float ident[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const float simple[16] = {2.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f,
                                  0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 2.f};
        const float x[8] = {0.1f, 0.2f, 0.3f, 0.4f, -0.2f, 0.3f, -0.4f, 0.5f};
        const int rows[2] = {1, 1};
        Tensor *gates[2] = {}, *ups[2] = {}, *downs[2] = {};
        const bool coli_cu_eg_g0 = tensor_upload(&gates[0], ident, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_eg_g0);
        const bool coli_cu_eg_u0 = tensor_upload(&ups[0], ident, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_eg_u0);
        const bool coli_cu_eg_d0 = tensor_upload(&downs[0], ident, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_eg_d0);
        const bool coli_cu_eg_g1 = tensor_upload(&gates[1], simple, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_eg_g1);
        const bool coli_cu_eg_u1 = tensor_upload(&ups[1], simple, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_eg_u1);
        const bool coli_cu_eg_d1 = tensor_upload(&downs[1], simple, nullptr, 0, D, D, 0, 0);
        CHECK(coli_cu_eg_d1);
        float y[8] = {};
        const bool coli_cu_eg = expert_group(gates, ups, downs, rows, 2, y, x);
        CHECK(coli_cu_eg);
        const bool coli_cu_eg_fin = all_finite(y, 8);
        CHECK(coli_cu_eg_fin);
        const bool coli_cu_eg_iss = expert_group_issue(gates, ups, downs, rows, 2, x);
        CHECK(coli_cu_eg_iss);
        const float *taken = expert_group_take(0);
        const bool coli_cu_eg_take = taken != nullptr;
        CHECK(coli_cu_eg_take);
        if (taken) {
            for (int i = 0; i < 8; ++i)
                CHECK_NEAR(taken[i], y[i], 1e-4);
        }
        tensor_free(gates[0]);
        tensor_free(ups[0]);
        tensor_free(downs[0]);
        tensor_free(gates[1]);
        tensor_free(ups[1]);
        tensor_free(downs[1]);
    }

    {
        const int H = 1, Q = 2, R = 2, V = 2, K = 2, T = 1;
        const float kvb_w[8] = {1.f, 0.f, 0.f, 1.f, 1.f, 0.f, 0.f, 1.f};
        const float q[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        const float latent[2] = {0.5f, 0.6f};
        const float rope[2] = {0.7f, 0.8f};
        float ctx[2] = {};
        Tensor *kvb = nullptr;
        const bool coli_cu_abs_up = tensor_upload(&kvb, kvb_w, nullptr, 0, K, H * (Q + V), 0, 0);
        CHECK(coli_cu_abs_up);
        const bool coli_cu_abs =
            attention_absorb(kvb, ctx, q, latent, rope, H, Q, R, V, K, T, 1.f);
        CHECK(coli_cu_abs);
        const bool coli_cu_abs_fin = all_finite(ctx, V);
        CHECK(coli_cu_abs_fin);
        // T=1 softmax is 1: ctx = Wv @ latent (identity Wv).
        CHECK_NEAR(ctx[0], latent[0], 1e-5);
        CHECK_NEAR(ctx[1], latent[1], 1e-5);
        tensor_free(kvb);
    }

    {
        const float ones[4] = {1.f, 1.f, 1.f, 1.f};
        const float weight[4] = {1.f, 1.f, 1.f, 1.f};
        float yn[4] = {};
        const bool coli_cu_rms = pipe_rmsnorm(0, yn, ones, weight, 1, 4, 1e-6f);
        CHECK(coli_cu_rms);
        const bool coli_cu_rms_fin = all_finite(yn, 4);
        CHECK(coli_cu_rms_fin);
        float gate[4] = {1.f, 1.f, 1.f, 1.f};
        const bool coli_cu_silu = pipe_silu_mul(0, gate, ones, 4);
        CHECK(coli_cu_silu);
        const bool coli_cu_silu_fin = all_finite(gate, 4);
        CHECK(coli_cu_silu_fin);
        float acc[4] = {1.f, 1.f, 1.f, 1.f};
        const bool coli_cu_add = pipe_add(0, acc, ones, 4);
        CHECK(coli_cu_add);
        CHECK_NEAR(acc[0], 2.f, 1e-5);
        CHECK_NEAR(acc[1], 2.f, 1e-5);
        CHECK_NEAR(acc[2], 2.f, 1e-5);
        CHECK_NEAR(acc[3], 2.f, 1e-5);
    }

    tensor_free(nullptr);
    shutdown();
    const bool coli_cu_reinit = init(nullptr, 0);
    CHECK(coli_cu_reinit);
    const bool coli_cu_reavail = available();
    CHECK(coli_cu_reavail);
    shutdown();
}

static void test_vk_ops_tier() {
    using namespace mvllm;
    using namespace mvllm::vk_ops;

    const bool vk_ops_init = init();
    CHECK(vk_ops_init);
    const bool vk_ops_avail = available();
    CHECK(vk_ops_avail);
    const char *vk_ops_bn = backend_name();
    CHECK(vk_ops_bn && std::strcmp(vk_ops_bn, "cpu") == 0);

    const float vk_ops_x[4] = {1.f, 0.f, 0.f, 0.f};
    const float vk_ops_w[4] = {1.f, 1.f, 1.f, 1.f};
    float vk_ops_y[4] = {};
    const bool vk_ops_rms = rmsnorm(vk_ops_y, vk_ops_x, vk_ops_w, 1, 4, 1e-6f);
    CHECK(vk_ops_rms);
    CHECK(std::isfinite(vk_ops_y[0]) && std::isfinite(vk_ops_y[1]));

    float vk_ops_acc[2] = {1.f, 2.f};
    const float vk_ops_addend[2] = {3.f, 4.f};
    const bool vk_ops_add = add(vk_ops_acc, vk_ops_addend, 2);
    CHECK(vk_ops_add);
    CHECK_NEAR(vk_ops_acc[0], 4.f, 1e-5);
    CHECK_NEAR(vk_ops_acc[1], 6.f, 1e-5);

    const float vk_ops_id[4] = {1.f, 0.f, 0.f, 1.f};
    const float vk_ops_in[2] = {0.5f, -0.25f};
    float vk_ops_out[2] = {};
    const bool vk_ops_g = gemm_f32(vk_ops_out, vk_ops_in, vk_ops_id, 1, 2, 2);
    CHECK(vk_ops_g);
    CHECK_NEAR(vk_ops_out[0], 0.5f, 1e-5);
    CHECK_NEAR(vk_ops_out[1], -0.25f, 1e-5);

    shutdown();
    const bool vk_ops_off = !available();
    CHECK(vk_ops_off);
}

static void test_metal_ops_tier() {
    using namespace mvllm;
    using namespace mvllm::metal_ops;

    const bool metal_ops_init = init();
    CHECK(metal_ops_init);
    const bool metal_ops_avail = available();
    CHECK(metal_ops_avail);

    {
        const float metal_ops_ones[4] = {1.f, 1.f, 1.f, 1.f};
        const float metal_ops_w[4] = {1.f, 1.f, 1.f, 1.f};
        float metal_ops_yn[4] = {};
        const bool metal_ops_rms = rmsnorm(metal_ops_yn, metal_ops_ones, metal_ops_w, 1, 4, 1e-6f);
        CHECK(metal_ops_rms);
        const bool metal_ops_rms_fin = std::isfinite(metal_ops_yn[0]) && std::isfinite(metal_ops_yn[1]) &&
                                       std::isfinite(metal_ops_yn[2]) && std::isfinite(metal_ops_yn[3]);
        CHECK(metal_ops_rms_fin);
    }

    {
        float metal_ops_acc[1] = {1.f};
        const float metal_ops_addend[1] = {2.f};
        const bool metal_ops_add = add(metal_ops_acc, metal_ops_addend, 1);
        CHECK(metal_ops_add);
        CHECK_NEAR(metal_ops_acc[0], 3.f, 1e-5);
    }

    {
        float metal_ops_g[4] = {0.5f, -0.25f, 1.f, -1.f};
        const float metal_ops_u[4] = {1.f, 2.f, 0.5f, -0.5f};
        const bool metal_ops_silu = silu_mul(metal_ops_g, metal_ops_u, 4);
        CHECK(metal_ops_silu);
        const bool metal_ops_silu_fin = std::isfinite(metal_ops_g[0]) && std::isfinite(metal_ops_g[1]) &&
                                        std::isfinite(metal_ops_g[2]) && std::isfinite(metal_ops_g[3]);
        CHECK(metal_ops_silu_fin);
    }

    {
        const int metal_ops_H = 2, metal_ops_hd = 4, metal_ops_K = 4, metal_ops_P = 8;
        std::vector<float> metal_ops_win_q(static_cast<size_t>(metal_ops_P) * metal_ops_K, 0.f);
        std::vector<float> metal_ops_win_k(static_cast<size_t>(metal_ops_P) * metal_ops_K, 0.f);
        std::vector<float> metal_ops_win_v(static_cast<size_t>(metal_ops_P) * metal_ops_K, 0.f);
        std::vector<float> metal_ops_qt(metal_ops_P, 0.1f);
        std::vector<float> metal_ops_kt(metal_ops_P, 0.2f);
        std::vector<float> metal_ops_tv(metal_ops_P, 0.3f);
        std::vector<float> metal_ops_taps_q(static_cast<size_t>(metal_ops_P) * metal_ops_K, 0.f);
        std::vector<float> metal_ops_taps_k(static_cast<size_t>(metal_ops_P) * metal_ops_K, 0.f);
        std::vector<float> metal_ops_taps_v(static_cast<size_t>(metal_ops_P) * metal_ops_K, 0.f);
        for (int p = 0; p < metal_ops_P; ++p) {
            metal_ops_taps_q[static_cast<size_t>(p) * metal_ops_K + (metal_ops_K - 1)] = 1.f;
            metal_ops_taps_k[static_cast<size_t>(p) * metal_ops_K + (metal_ops_K - 1)] = 1.f;
            metal_ops_taps_v[static_cast<size_t>(p) * metal_ops_K + (metal_ops_K - 1)] = 1.f;
        }
        std::vector<float> metal_ops_S(static_cast<size_t>(metal_ops_H) * metal_ops_hd * metal_ops_hd,
                                       0.f);
        std::vector<float> metal_ops_alpha(static_cast<size_t>(metal_ops_H) * metal_ops_hd, 1.f);
        std::vector<float> metal_ops_beta(metal_ops_H, 0.5f);
        std::vector<float> metal_ops_oh(static_cast<size_t>(metal_ops_H) * metal_ops_hd, 0.f);
        const bool metal_ops_kda =
            kda_fused_token(metal_ops_win_q.data(), metal_ops_qt.data(), metal_ops_win_k.data(),
                            metal_ops_kt.data(), metal_ops_win_v.data(), metal_ops_tv.data(),
                            metal_ops_taps_q.data(), metal_ops_taps_k.data(), metal_ops_taps_v.data(),
                            metal_ops_S.data(), metal_ops_alpha.data(), metal_ops_beta.data(),
                            metal_ops_oh.data(), metal_ops_P, metal_ops_K, metal_ops_H, metal_ops_hd);
        CHECK(metal_ops_kda);
        bool metal_ops_oh_ok = true;
        for (float v : metal_ops_oh)
            metal_ops_oh_ok = metal_ops_oh_ok && std::isfinite(v);
        const bool metal_ops_kda_fin = metal_ops_oh_ok;
        CHECK(metal_ops_kda_fin);
    }

    {
        float metal_ops_ld_x[4] = {1.f, 0.f, 0.f, 0.f};
        const float metal_ops_ld_attn[4] = {0.f, 1.f, 0.f, 0.f};
        const float metal_ops_ld_post[4] = {1.f, 1.f, 1.f, 1.f};
        const float metal_ops_ld_id[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                           0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        float metal_ops_ld_nrm[4] = {};
        float metal_ops_ld_sh[4] = {};
        const bool metal_ops_ld =
            layer_decode(metal_ops_ld_x, metal_ops_ld_attn, metal_ops_ld_post, metal_ops_ld_id,
                         metal_ops_ld_id, metal_ops_ld_id, 4, 4, 1e-6f, metal_ops_ld_nrm,
                         metal_ops_ld_sh);
        CHECK(metal_ops_ld);
        const bool metal_ops_ld_nrm_fin =
            std::isfinite(metal_ops_ld_nrm[0]) && std::isfinite(metal_ops_ld_nrm[1]) &&
            std::isfinite(metal_ops_ld_nrm[2]) && std::isfinite(metal_ops_ld_nrm[3]);
        CHECK(metal_ops_ld_nrm_fin);
        const bool metal_ops_ld_resid = (metal_ops_ld_x[0] != 0.f) && (metal_ops_ld_x[1] != 0.f);
        CHECK(metal_ops_ld_resid);
        const bool metal_ops_ld_sh_fin =
            std::isfinite(metal_ops_ld_sh[0]) && std::isfinite(metal_ops_ld_sh[1]) &&
            std::isfinite(metal_ops_ld_sh[2]) && std::isfinite(metal_ops_ld_sh[3]);
        CHECK(metal_ops_ld_sh_fin);
    }

    {
        const float metal_ops_moe_id[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                            0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const void *metal_ops_moe_g[1] = {metal_ops_moe_id};
        const void *metal_ops_moe_u[1] = {metal_ops_moe_id};
        const void *metal_ops_moe_d[1] = {metal_ops_moe_id};
        const float metal_ops_moe_x[4] = {0.5f, -0.25f, 1.f, -1.f};
        const int metal_ops_moe_xoff[1] = {0};
        const int metal_ops_moe_nr[1] = {1};
        const int metal_ops_moe_rows[1] = {0};
        const float metal_ops_moe_rw[1] = {1.f};
        float metal_ops_moe_out[4] = {};
        const bool metal_ops_moe =
            moe_block(1, 4, 4, 0, 0, metal_ops_moe_g, metal_ops_moe_u, metal_ops_moe_d, nullptr,
                      nullptr, nullptr, metal_ops_moe_x, metal_ops_moe_xoff, metal_ops_moe_nr,
                      metal_ops_moe_rows, metal_ops_moe_rw, metal_ops_moe_out, 1);
        CHECK(metal_ops_moe);
        const bool metal_ops_moe_out_fin =
            std::isfinite(metal_ops_moe_out[0]) && std::isfinite(metal_ops_moe_out[1]) &&
            std::isfinite(metal_ops_moe_out[2]) && std::isfinite(metal_ops_moe_out[3]);
        CHECK(metal_ops_moe_out_fin);
        float metal_ops_moe_exp[4];
        for (int metal_ops_moe_i = 0; metal_ops_moe_i < 4; ++metal_ops_moe_i) {
            const float metal_ops_moe_v = metal_ops_moe_x[metal_ops_moe_i];
            const float metal_ops_moe_sig =
                metal_ops_moe_v >= 0.f
                    ? 1.f / (1.f + std::exp(-metal_ops_moe_v))
                    : (std::exp(metal_ops_moe_v) / (1.f + std::exp(metal_ops_moe_v)));
            metal_ops_moe_exp[metal_ops_moe_i] =
                (metal_ops_moe_v * metal_ops_moe_sig) * metal_ops_moe_x[metal_ops_moe_i];
        }
        CHECK_NEAR(metal_ops_moe_out[0], metal_ops_moe_exp[0], 1e-4);
        CHECK_NEAR(metal_ops_moe_out[1], metal_ops_moe_exp[1], 1e-4);
        CHECK_NEAR(metal_ops_moe_out[2], metal_ops_moe_exp[2], 1e-4);
        CHECK_NEAR(metal_ops_moe_out[3], metal_ops_moe_exp[3], 1e-4);
    }

    {
        float metal_ops_ldf_x[4] = {1.f, 0.f, 0.f, 0.f};
        const float metal_ops_ldf_attn[4] = {0.f, 1.f, 0.f, 0.f};
        const float metal_ops_ldf_post[4] = {1.f, 1.f, 1.f, 1.f};
        const float metal_ops_ldf_id[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                            0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const float metal_ops_ldf_rw[8] = {0.2f, 0.1f, 0.1f, 0.1f, 0.3f, 0.1f, 0.1f, 0.1f};
        float metal_ops_ldf_nrm[4] = {};
        float metal_ops_ldf_sh[4] = {};
        int metal_ops_ldf_idx[2] = {-1, -1};
        float metal_ops_ldf_w[2] = {};
        const bool metal_ops_ldf = layer_decode_full(
            metal_ops_ldf_x, metal_ops_ldf_attn, nullptr, metal_ops_ldf_post, metal_ops_ldf_id,
            metal_ops_ldf_id, metal_ops_ldf_id, 4, 4, 1e-6f, Act::Situ, 4.f, 25.f,
            metal_ops_ldf_rw, nullptr, 2, 2, 1.f, nullptr, metal_ops_ldf_nrm, metal_ops_ldf_sh,
            metal_ops_ldf_idx, metal_ops_ldf_w);
        CHECK(metal_ops_ldf);
        CHECK(std::isfinite(metal_ops_ldf_sh[0]) && std::isfinite(metal_ops_ldf_sh[1]));
        CHECK(metal_ops_ldf_idx[0] >= 0 && metal_ops_ldf_idx[0] < 2);
        CHECK(metal_ops_ldf_w[0] >= 0.f);
    }

    {
        const float metal_ops_moe_id[16] = {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
                                            0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
        const void *metal_ops_moe_g[1] = {metal_ops_moe_id};
        const void *metal_ops_moe_u[1] = {metal_ops_moe_id};
        const void *metal_ops_moe_d[1] = {metal_ops_moe_id};
        const float metal_ops_moe_x[4] = {0.5f, -0.25f, 1.f, -1.f};
        const int metal_ops_moe_xoff[1] = {0};
        const int metal_ops_moe_nr[1] = {1};
        const int metal_ops_moe_rows[1] = {0};
        const float metal_ops_moe_rw[1] = {1.f};
        float metal_ops_moe_out[4] = {};
        const bool metal_ops_moe_c =
            moe_block(1, 4, 4, 0, 0, metal_ops_moe_g, metal_ops_moe_u, metal_ops_moe_d, nullptr,
                      nullptr, nullptr, metal_ops_moe_x, metal_ops_moe_xoff, metal_ops_moe_nr,
                      metal_ops_moe_rows, metal_ops_moe_rw, metal_ops_moe_out, 1, Act::ClampSwiGLU,
                      10.f, 0.f);
        CHECK(metal_ops_moe_c);
        CHECK(std::isfinite(metal_ops_moe_out[0]) && std::isfinite(metal_ops_moe_out[1]));
        float metal_ops_moe_ref[4];
        for (int i = 0; i < 4; ++i)
            metal_ops_moe_ref[i] = mvllm::quant::clamped_swiglu(metal_ops_moe_x[i], metal_ops_moe_x[i], 10.f);
        CHECK_NEAR(metal_ops_moe_out[0], metal_ops_moe_ref[0], 1e-4);
        CHECK_NEAR(metal_ops_moe_out[1], metal_ops_moe_ref[1], 1e-4);
        CHECK_NEAR(metal_ops_moe_out[2], metal_ops_moe_ref[2], 1e-4);
        CHECK_NEAR(metal_ops_moe_out[3], metal_ops_moe_ref[3], 1e-4);
    }

    {
        const int metal_ops_ldk_H = 2, metal_ops_ldk_hd = 4, metal_ops_ldk_K = 4, metal_ops_ldk_P = 8;
        std::vector<float> metal_ops_ldk_wq(static_cast<size_t>(metal_ops_ldk_P) * metal_ops_ldk_K, 0.f);
        std::vector<float> metal_ops_ldk_wk(static_cast<size_t>(metal_ops_ldk_P) * metal_ops_ldk_K, 0.f);
        std::vector<float> metal_ops_ldk_wv(static_cast<size_t>(metal_ops_ldk_P) * metal_ops_ldk_K, 0.f);
        std::vector<float> metal_ops_ldk_qt(metal_ops_ldk_P, 0.1f);
        std::vector<float> metal_ops_ldk_kt(metal_ops_ldk_P, 0.2f);
        std::vector<float> metal_ops_ldk_tv(metal_ops_ldk_P, 0.3f);
        std::vector<float> metal_ops_ldk_tq(static_cast<size_t>(metal_ops_ldk_P) * metal_ops_ldk_K, 0.f);
        std::vector<float> metal_ops_ldk_tk(static_cast<size_t>(metal_ops_ldk_P) * metal_ops_ldk_K, 0.f);
        std::vector<float> metal_ops_ldk_tvp(static_cast<size_t>(metal_ops_ldk_P) * metal_ops_ldk_K, 0.f);
        for (int p = 0; p < metal_ops_ldk_P; ++p) {
            metal_ops_ldk_tq[static_cast<size_t>(p) * metal_ops_ldk_K + (metal_ops_ldk_K - 1)] = 1.f;
            metal_ops_ldk_tk[static_cast<size_t>(p) * metal_ops_ldk_K + (metal_ops_ldk_K - 1)] = 1.f;
            metal_ops_ldk_tvp[static_cast<size_t>(p) * metal_ops_ldk_K + (metal_ops_ldk_K - 1)] = 1.f;
        }
        std::vector<float> metal_ops_ldk_S(
            static_cast<size_t>(metal_ops_ldk_H) * metal_ops_ldk_hd * metal_ops_ldk_hd, 0.f);
        std::vector<float> metal_ops_ldk_al(static_cast<size_t>(metal_ops_ldk_H) * metal_ops_ldk_hd, 1.f);
        std::vector<float> metal_ops_ldk_be(metal_ops_ldk_H, 0.5f);
        std::vector<float> metal_ops_ldk_oh(static_cast<size_t>(metal_ops_ldk_P), 0.f);
        KdaToken metal_ops_ldk_tok{};
        metal_ops_ldk_tok.win_q = metal_ops_ldk_wq.data();
        metal_ops_ldk_tok.qt = metal_ops_ldk_qt.data();
        metal_ops_ldk_tok.win_k = metal_ops_ldk_wk.data();
        metal_ops_ldk_tok.kt = metal_ops_ldk_kt.data();
        metal_ops_ldk_tok.win_v = metal_ops_ldk_wv.data();
        metal_ops_ldk_tok.tv = metal_ops_ldk_tv.data();
        metal_ops_ldk_tok.taps_q = metal_ops_ldk_tq.data();
        metal_ops_ldk_tok.taps_k = metal_ops_ldk_tk.data();
        metal_ops_ldk_tok.taps_v = metal_ops_ldk_tvp.data();
        metal_ops_ldk_tok.S = metal_ops_ldk_S.data();
        metal_ops_ldk_tok.alpha = metal_ops_ldk_al.data();
        metal_ops_ldk_tok.beta = metal_ops_ldk_be.data();
        metal_ops_ldk_tok.oh = metal_ops_ldk_oh.data();
        metal_ops_ldk_tok.P = metal_ops_ldk_P;
        metal_ops_ldk_tok.K = metal_ops_ldk_K;
        metal_ops_ldk_tok.H = metal_ops_ldk_H;
        metal_ops_ldk_tok.hd = metal_ops_ldk_hd;
        float metal_ops_ldk_x[8] = {1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const float metal_ops_ldk_post[8] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        float metal_ops_ldk_nrm[8] = {};
        const bool metal_ops_ldk =
            layer_decode_kda(metal_ops_ldk_tok, metal_ops_ldk_x, nullptr, metal_ops_ldk_post, nullptr,
                             nullptr, nullptr, 8, 0, 1e-6f, Act::Silu, 0.f, 0.f, nullptr, nullptr, 0,
                             0, 1.f, nullptr, metal_ops_ldk_nrm, nullptr, nullptr, nullptr);
        CHECK(metal_ops_ldk);
        bool metal_ops_ldk_fin = std::isfinite(metal_ops_ldk_nrm[0]) && std::isfinite(metal_ops_ldk_oh[0]);
        for (int i = 1; i < 8; ++i)
            metal_ops_ldk_fin = metal_ops_ldk_fin && std::isfinite(metal_ops_ldk_nrm[i]) &&
                                std::isfinite(metal_ops_ldk_oh[i]);
        CHECK(metal_ops_ldk_fin);
        CHECK(metal_ops_ldk_x[0] != 0.f);
    }

    {
        const int metal_ops_ldm_H = 1, metal_ops_ldm_QK = 2, metal_ops_ldm_R = 0, metal_ops_ldm_V = 2,
                  metal_ops_ldm_L = 2, metal_ops_ldm_D = 2;
        const float metal_ops_ldm_q[2] = {0.5f, -0.25f};
        const float metal_ops_ldm_c[2] = {1.f, 0.f};
        const float metal_ops_ldm_kt[4] = {1.f, 0.f, 0.f, 1.f};
        const float metal_ops_ldm_v[4] = {1.f, 0.f, 0.f, 1.f};
        const float metal_ops_ldm_o[4] = {1.f, 0.f, 0.f, 1.f};
        float metal_ops_ldm_oh[2] = {};
        MlaAbsorb metal_ops_ldm_abs{};
        metal_ops_ldm_abs.q = metal_ops_ldm_q;
        metal_ops_ldm_abs.cache = metal_ops_ldm_c;
        metal_ops_ldm_abs.w_kt = metal_ops_ldm_kt;
        metal_ops_ldm_abs.w_v = metal_ops_ldm_v;
        metal_ops_ldm_abs.w_o = metal_ops_ldm_o;
        metal_ops_ldm_abs.oh = metal_ops_ldm_oh;
        metal_ops_ldm_abs.H = metal_ops_ldm_H;
        metal_ops_ldm_abs.QK = metal_ops_ldm_QK;
        metal_ops_ldm_abs.R = metal_ops_ldm_R;
        metal_ops_ldm_abs.Vh = metal_ops_ldm_V;
        metal_ops_ldm_abs.L = metal_ops_ldm_L;
        metal_ops_ldm_abs.stride = metal_ops_ldm_L + metal_ops_ldm_R;
        metal_ops_ldm_abs.T = 1;
        float metal_ops_ldm_x[2] = {0.f, 0.f};
        const float metal_ops_ldm_post[2] = {1.f, 1.f};
        float metal_ops_ldm_nrm[2] = {};
        const bool metal_ops_ldm =
            layer_decode_mla(metal_ops_ldm_abs, metal_ops_ldm_x, metal_ops_ldm_post, metal_ops_ldm_D,
                             1e-6f, metal_ops_ldm_nrm);
        CHECK(metal_ops_ldm);
        CHECK(std::isfinite(metal_ops_ldm_nrm[0]) && std::isfinite(metal_ops_ldm_nrm[1]));
        CHECK(std::isfinite(metal_ops_ldm_oh[0]) && std::isfinite(metal_ops_ldm_x[0]));
    }

    {
        const int metal_ops_mx_I = 32, metal_ops_mx_O = 32;
        std::vector<float> metal_ops_mx_wf(static_cast<size_t>(metal_ops_mx_O) * metal_ops_mx_I, 0.f);
        for (int i = 0; i < metal_ops_mx_I; ++i)
            metal_ops_mx_wf[static_cast<size_t>(i) * metal_ops_mx_I + i] = 0.5f;
        const int metal_ops_mx_pb = metal_ops_mx_O * ((metal_ops_mx_I + 1) / 2);
        const int metal_ops_mx_sb = metal_ops_mx_O * ((metal_ops_mx_I + 31) / 32);
        std::vector<uint8_t> metal_ops_mx_gp(static_cast<size_t>(metal_ops_mx_pb));
        std::vector<uint8_t> metal_ops_mx_up(static_cast<size_t>(metal_ops_mx_pb));
        std::vector<uint8_t> metal_ops_mx_dp(static_cast<size_t>(metal_ops_mx_pb));
        std::vector<uint8_t> metal_ops_mx_gs(static_cast<size_t>(metal_ops_mx_sb));
        std::vector<uint8_t> metal_ops_mx_us(static_cast<size_t>(metal_ops_mx_sb));
        std::vector<uint8_t> metal_ops_mx_ds(static_cast<size_t>(metal_ops_mx_sb));
        mvllm::quant::pack_mxfp4(metal_ops_mx_wf.data(), metal_ops_mx_O, metal_ops_mx_I,
                                 metal_ops_mx_gp.data(), metal_ops_mx_gs.data());
        mvllm::quant::pack_mxfp4(metal_ops_mx_wf.data(), metal_ops_mx_O, metal_ops_mx_I,
                                 metal_ops_mx_up.data(), metal_ops_mx_us.data());
        mvllm::quant::pack_mxfp4(metal_ops_mx_wf.data(), metal_ops_mx_I, metal_ops_mx_O,
                                 metal_ops_mx_dp.data(), metal_ops_mx_ds.data());
        const void *metal_ops_mx_g[1] = {metal_ops_mx_gp.data()};
        const void *metal_ops_mx_u[1] = {metal_ops_mx_up.data()};
        const void *metal_ops_mx_d[1] = {metal_ops_mx_dp.data()};
        const float *metal_ops_mx_gsp[1] = {reinterpret_cast<const float *>(metal_ops_mx_gs.data())};
        const float *metal_ops_mx_usp[1] = {reinterpret_cast<const float *>(metal_ops_mx_us.data())};
        const float *metal_ops_mx_dsp[1] = {reinterpret_cast<const float *>(metal_ops_mx_ds.data())};
        std::vector<float> metal_ops_mx_x(static_cast<size_t>(metal_ops_mx_I), 0.f);
        metal_ops_mx_x[0] = 1.f;
        const int metal_ops_mx_xoff[1] = {0};
        const int metal_ops_mx_nr[1] = {1};
        const int metal_ops_mx_rows[1] = {0};
        const float metal_ops_mx_rw[1] = {1.f};
        std::vector<float> metal_ops_mx_out(static_cast<size_t>(metal_ops_mx_I), 0.f);
        const bool metal_ops_mx =
            moe_block(1, metal_ops_mx_I, metal_ops_mx_O, 7, 32, metal_ops_mx_g, metal_ops_mx_u,
                      metal_ops_mx_d, metal_ops_mx_gsp, metal_ops_mx_usp, metal_ops_mx_dsp,
                      metal_ops_mx_x.data(), metal_ops_mx_xoff, metal_ops_mx_nr, metal_ops_mx_rows,
                      metal_ops_mx_rw, metal_ops_mx_out.data(), 1, Act::Situ, 4.f, 25.f);
        CHECK(metal_ops_mx);
        bool metal_ops_mx_fin = true;
        float metal_ops_mx_e = 0.f;
        for (float v : metal_ops_mx_out) {
            metal_ops_mx_fin = metal_ops_mx_fin && std::isfinite(v);
            metal_ops_mx_e += v * v;
        }
        CHECK(metal_ops_mx_fin);
        CHECK(metal_ops_mx_e > 0.f);
    }
}

static void test_metal_h3_tier() {
    using namespace mvllm;
    using namespace mvllm::metal_h3;

    const bool metal_h3_init = init();
    CHECK(metal_h3_init);

    const int metal_h3_hidden = 8, metal_h3_inner = 8, metal_h3_ffn = 8, metal_h3_hd = 4,
              metal_h3_tokens = 2;
    const int64_t metal_h3_qkv_n = static_cast<int64_t>(3) * metal_h3_inner * metal_h3_hidden;
    const int64_t metal_h3_out_n = static_cast<int64_t>(metal_h3_hidden) * metal_h3_inner;
    const int64_t metal_h3_fc1_n = static_cast<int64_t>(2) * metal_h3_ffn * metal_h3_hidden;
    const int64_t metal_h3_fc2_n = static_cast<int64_t>(metal_h3_hidden) * metal_h3_ffn;
    std::vector<uint8_t> metal_h3_blob(
        static_cast<size_t>(2 * (metal_h3_qkv_n + metal_h3_out_n + metal_h3_fc1_n + metal_h3_fc2_n)));
    uint16_t *metal_h3_bf = reinterpret_cast<uint16_t *>(metal_h3_blob.data());
    for (int64_t i = 0; i < metal_h3_qkv_n + metal_h3_out_n + metal_h3_fc1_n + metal_h3_fc2_n; ++i)
        metal_h3_bf[i] = bf16_encode(((i * 17) % 11 - 5) * 0.05f);
    std::vector<float> metal_h3_x(static_cast<size_t>(metal_h3_tokens) * metal_h3_hidden);
    for (int i = 0; i < metal_h3_tokens * metal_h3_hidden; ++i)
        metal_h3_x[static_cast<size_t>(i)] = ((i % 7) - 3) * 0.1f;
    const bool metal_h3_dit =
        dit_residual(metal_h3_blob.data(), metal_h3_qkv_n * 2, metal_h3_out_n * 2, metal_h3_fc1_n * 2,
                     metal_h3_fc2_n * 2, metal_h3_hidden, metal_h3_inner, metal_h3_ffn, metal_h3_hd,
                     metal_h3_x.data(), metal_h3_tokens, 1e-6f, nullptr, nullptr, nullptr, nullptr,
                     nullptr);
    CHECK(metal_h3_dit);
    bool metal_h3_x_ok = true;
    for (float v : metal_h3_x)
        metal_h3_x_ok = metal_h3_x_ok && std::isfinite(v);
    const bool metal_h3_dit_fin = metal_h3_x_ok;
    CHECK(metal_h3_dit_fin);

    {
        const int8_t metal_h3_iw[4] = {1, -1, 2, 0};
        const float metal_h3_is[2] = {0.5f, 0.25f};
        const float metal_h3_ix[2] = {1.f, 2.f};
        float metal_h3_iy[2] = {};
        const bool metal_h3_i8 = gemm_int8(metal_h3_iy, metal_h3_ix, metal_h3_iw, metal_h3_is, 1, 2, 2);
        CHECK(metal_h3_i8);
        CHECK_NEAR(metal_h3_iy[0], (1.f * 1 + 2.f * -1) * 0.5f, 1e-5);
        CHECK_NEAR(metal_h3_iy[1], (1.f * 2 + 2.f * 0) * 0.25f, 1e-5);
    }
    {
        const float metal_h3_up[8] = {1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f};
        const float metal_h3_dn[8] = {1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 1.f, 0.f};
        float metal_h3_nx[4] = {0.2f, -0.1f, 0.3f, 0.f};
        float metal_h3_ny[4] = {0.f, 0.f, 0.f, 0.f};
        const bool metal_h3_nax = nax_mlp(metal_h3_ny, metal_h3_nx, metal_h3_up, metal_h3_dn, 1, 4, 2);
        CHECK(metal_h3_nax);
        CHECK(std::isfinite(metal_h3_ny[0]) && std::isfinite(metal_h3_ny[1]));
    }
    {
        float metal_h3_vx[4] = {1.f, 0.f, 0.f, 0.f};
        const float metal_h3_sk[4] = {0.f, 1.f, 0.f, 0.f};
        const float metal_h3_vw[4] = {1.f, 1.f, 1.f, 1.f};
        float metal_h3_vy[4] = {};
        const bool metal_h3_vae = vae_rms_add(metal_h3_vx, metal_h3_sk, metal_h3_vw, metal_h3_vy, 4, 1e-6f);
        CHECK(metal_h3_vae);
        CHECK(metal_h3_vx[0] != 0.f && metal_h3_vx[1] != 0.f);
        CHECK(std::isfinite(metal_h3_vy[0]));
    }

    {
        const int metal_h3_vb_rows = 2, metal_h3_vb_H = 4, metal_h3_vb_heads = 2, metal_h3_vb_hd = 2,
                  metal_h3_vb_I = 4;
        std::vector<float> metal_h3_vb_x(static_cast<size_t>(metal_h3_vb_rows) * metal_h3_vb_H, 0.1f);
        metal_h3_vb_x[0] = 1.f;
        std::vector<float> metal_h3_vb_qkv(static_cast<size_t>(3 * metal_h3_vb_H) * metal_h3_vb_H, 0.f);
        std::vector<float> metal_h3_vb_proj(static_cast<size_t>(metal_h3_vb_H) * metal_h3_vb_H, 0.f);
        std::vector<float> metal_h3_vb_fc1(static_cast<size_t>(metal_h3_vb_I) * metal_h3_vb_H, 0.f);
        std::vector<float> metal_h3_vb_fc2(static_cast<size_t>(metal_h3_vb_H) * metal_h3_vb_I, 0.f);
        for (int i = 0; i < metal_h3_vb_H; ++i) {
            metal_h3_vb_proj[static_cast<size_t>(i) * metal_h3_vb_H + i] = 1.f;
            metal_h3_vb_qkv[static_cast<size_t>(i) * metal_h3_vb_H + i] = 0.2f;
            metal_h3_vb_fc1[static_cast<size_t>(i) * metal_h3_vb_H + i] = 0.2f;
            metal_h3_vb_fc2[static_cast<size_t>(i) * metal_h3_vb_I + i] = 0.2f;
        }
        const bool metal_h3_vb = vision_block(
            metal_h3_vb_x.data(), metal_h3_vb_rows, metal_h3_vb_H, metal_h3_vb_heads, metal_h3_vb_hd,
            metal_h3_vb_I, nullptr, nullptr, metal_h3_vb_qkv.data(), nullptr, metal_h3_vb_proj.data(),
            nullptr, nullptr, nullptr, metal_h3_vb_fc1.data(), nullptr, metal_h3_vb_fc2.data(),
            nullptr, nullptr, nullptr, 0, 1e-6f);
        CHECK(metal_h3_vb);
        CHECK(std::isfinite(metal_h3_vb_x[0]) && std::isfinite(metal_h3_vb_x[1]));
    }

    {
        const int metal_h3_ap_B = 1, metal_h3_ap_L = 2, metal_h3_ap_C = 4, metal_h3_ap_ch = 4,
                  metal_h3_ap_heads = 2;
        std::vector<float> metal_h3_ap_seq(static_cast<size_t>(metal_h3_ap_B * metal_h3_ap_L) *
                                               metal_h3_ap_C,
                                           0.1f);
        metal_h3_ap_seq[0] = 1.f;
        std::vector<float> metal_h3_ap_base(metal_h3_ap_seq.size(), 0.f);
        std::vector<float> metal_h3_ap_qkv(static_cast<size_t>(3 * metal_h3_ap_C) * metal_h3_ap_C, 0.f);
        std::vector<float> metal_h3_ap_proj(static_cast<size_t>(metal_h3_ap_ch) * metal_h3_ap_ch, 0.f);
        std::vector<float> metal_h3_ap_w0(static_cast<size_t>(2 * metal_h3_ap_ch) * metal_h3_ap_ch, 0.f);
        std::vector<float> metal_h3_ap_w1(static_cast<size_t>(2 * metal_h3_ap_ch) * metal_h3_ap_ch, 0.f);
        std::vector<float> metal_h3_ap_w2(static_cast<size_t>(metal_h3_ap_ch) * (2 * metal_h3_ap_ch),
                                          0.f);
        for (int i = 0; i < metal_h3_ap_C; ++i) {
            metal_h3_ap_qkv[static_cast<size_t>(i) * metal_h3_ap_C + i] = 0.2f;
            metal_h3_ap_proj[static_cast<size_t>(i) * metal_h3_ap_ch + i] = 1.f;
        }
        const bool metal_h3_ap = audio_pre_block(
            metal_h3_ap_base.data(), metal_h3_ap_seq.data(), metal_h3_ap_B, metal_h3_ap_L,
            metal_h3_ap_C, metal_h3_ap_ch, metal_h3_ap_heads, nullptr, nullptr,
            metal_h3_ap_qkv.data(), nullptr, nullptr, nullptr, metal_h3_ap_proj.data(), nullptr,
            nullptr, nullptr, nullptr, nullptr, metal_h3_ap_w0.data(), nullptr,
            metal_h3_ap_w1.data(), nullptr, metal_h3_ap_w2.data(), nullptr, 1e-6f);
        CHECK(metal_h3_ap);
        CHECK(std::isfinite(metal_h3_ap_base[0]) && std::isfinite(metal_h3_ap_base[1]));
    }

    {
        const int metal_h3_vt_T = 2, metal_h3_vt_H = 4, metal_h3_vt_heads = 2, metal_h3_vt_hd = 2,
                  metal_h3_vt_ffn = 4;
        std::vector<float> metal_h3_vt_x(static_cast<size_t>(metal_h3_vt_T) * metal_h3_vt_H, 0.1f);
        metal_h3_vt_x[0] = 1.f;
        std::vector<float> metal_h3_vt_qkv(static_cast<size_t>(3 * metal_h3_vt_H) * metal_h3_vt_H, 0.f);
        std::vector<float> metal_h3_vt_out(static_cast<size_t>(metal_h3_vt_H) * metal_h3_vt_H, 0.f);
        std::vector<float> metal_h3_vt_w1(static_cast<size_t>(2 * metal_h3_vt_ffn) * metal_h3_vt_H, 0.f);
        std::vector<float> metal_h3_vt_w2(static_cast<size_t>(metal_h3_vt_H) * metal_h3_vt_ffn, 0.f);
        for (int i = 0; i < metal_h3_vt_H; ++i) {
            metal_h3_vt_qkv[static_cast<size_t>(i) * metal_h3_vt_H + i] = 0.2f;
            metal_h3_vt_out[static_cast<size_t>(i) * metal_h3_vt_H + i] = 1.f;
            metal_h3_vt_w1[static_cast<size_t>(i) * metal_h3_vt_H + i] = 0.2f;
            metal_h3_vt_w2[static_cast<size_t>(i) * metal_h3_vt_ffn + i] = 0.2f;
        }
        const bool metal_h3_vt = vae_transformer_block(
            metal_h3_vt_x.data(), metal_h3_vt_T, metal_h3_vt_H, metal_h3_vt_heads, metal_h3_vt_hd,
            nullptr, metal_h3_vt_qkv.data(), nullptr, metal_h3_vt_out.data(), nullptr, nullptr,
            nullptr, metal_h3_vt_w1.data(), nullptr, metal_h3_vt_w2.data(), nullptr, nullptr,
            metal_h3_vt_ffn, 1e-6f);
        CHECK(metal_h3_vt);
        CHECK(std::isfinite(metal_h3_vt_x[0]) && std::isfinite(metal_h3_vt_x[1]));
    }
}

static void test_llama_dims_helpers() {
    const bool llama_dims_nlayers = N_LAYERS == 16;
    CHECK(llama_dims_nlayers);
    const bool llama_dims_embed = EMBEDDING_LENGTH == 2048;
    CHECK(llama_dims_embed);
    const bool llama_dims_hidden = HIDDEN_DIM == 8192;
    CHECK(llama_dims_hidden);
    const bool llama_dims_vocab = VOCAB_SIZE == 128256;
    CHECK(llama_dims_vocab);
    const bool llama_dims_qh = NUM_Q_HEADS == 32;
    CHECK(llama_dims_qh);
    const bool llama_dims_kh = NUM_K_HEADS == 8;
    CHECK(llama_dims_kh);
    const bool llama_dims_gqa = GQA_Q_TO_K_RATIO == 4;
    CHECK(llama_dims_gqa);
    const bool llama_dims_maxp = DEFAULT_MAX_PROMPT == 512;
    CHECK(llama_dims_maxp);
    const bool llama_dims_clamp0 = llama_clamp_n(0, 1) == 1;
    CHECK(llama_dims_clamp0);
    const bool llama_dims_clamp8 = llama_clamp_n(8, 1) == 8;
    CHECK(llama_dims_clamp8);
    const bool llama_dims_bidx =
        llama_block_index(1, 2, 3) == 1 * N_LAYERS * MAX_BLOCKS_PER_SEQ + 2 * MAX_BLOCKS_PER_SEQ + 3;
    CHECK(llama_dims_bidx);
    const bool llama_dims_slot = llama_slot_table_elems() == N_LAYERS * MAX_BLOCKS_PER_SEQ;
    CHECK(llama_dims_slot);
    const bool llama_dims_btab = llama_block_table_elems(2) == 2 * llama_slot_table_elems();
    CHECK(llama_dims_btab);
}

static void test_idot() {
    using namespace mvllm::quant;
    const int I = 32, O = 4;
    std::vector<float> W(static_cast<size_t>(O) * I), X(I), Ya(O), Yb(O);
    for (int i = 0; i < O * I; ++i)
        W[static_cast<size_t>(i)] = ((i * 17) % 11 - 5) * 0.15f;
    for (int i = 0; i < I; ++i)
        X[static_cast<size_t>(i)] = ((i * 5) % 9 - 4) * 0.25f;
    std::vector<uint8_t> packed(static_cast<size_t>((O * I + 1) / 2));
    std::vector<uint8_t> scales(static_cast<size_t>(O) * ((I + 31) / 32));
    pack_mxfp4(W.data(), O, I, packed.data(), scales.data());
    matmul_mxfp4(Ya.data(), X.data(), packed.data(), scales.data(), 1, I, O);
    matmul_mxfp4_i8(Yb.data(), X.data(), packed.data(), scales.data(), 1, I, O);

    for (int o = 0; o < O; ++o)
        CHECK(std::isfinite(Ya[o]) && std::isfinite(Yb[o]));

    int ia = 0;
    for (int o = 1; o < O; ++o) {
        if (std::fabs(Ya[o]) > std::fabs(Ya[ia]))
            ia = o;
    }
    bool same_sign = (Ya[ia] == 0.f && Yb[ia] == 0.f) || (Ya[ia] * Yb[ia] > 0.f);
    float max_rel = 0.f;
    for (int o = 0; o < O; ++o) {
        float d = std::max(std::fabs(Ya[o]), 1e-6f);
        max_rel = std::max(max_rel, std::fabs(Ya[o] - Yb[o]) / d);
    }
    CHECK(same_sign || max_rel < 0.3f);

    // Remainder I falls back to the float kernel and must stay finite.
    const int I2 = 40;
    std::vector<float> W2(static_cast<size_t>(O) * I2, 0.1f), X2(I2, 0.2f), Y2(O);
    std::vector<uint8_t> p2(static_cast<size_t>((O * I2 + 1) / 2));
    std::vector<uint8_t> s2(static_cast<size_t>(O) * ((I2 + 31) / 32));
    pack_mxfp4(W2.data(), O, I2, p2.data(), s2.data());
    matmul_mxfp4_i8(Y2.data(), X2.data(), p2.data(), s2.data(), 1, I2, O);
    for (int o = 0; o < O; ++o)
        CHECK(std::isfinite(Y2[o]));
}

static void test_offload_generate() {
    using namespace mvllm;
    std::string err;

    std::string kdir = tmpdir();
    write_file(kdir + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 3,
      "vocab_size": 48,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_attention_heads": 4,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 8,
      "v_head_dim": 8,
      "num_experts": 4,
      "num_experts_per_token": 2,
      "moe_intermediate_size": 16,
      "routed_expert_hidden_size": 16,
      "num_shared_experts": 1,
      "attn_res_block_size": 2,
      "rms_norm_eps": 1e-5,
      "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "kda_layers": [1, 2],
        "full_attn_layers": [3],
        "use_full_rank_gate": true,
        "gate_lower_bound": -5.0
      },
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    Engine ek;
    RuntimeConfig rt;
    rt.expert_gb = 0.001;
    CHECK(ek.load(kdir, rt, err) == Status::Ok);
    GenParams gp;
    gp.max_new_tokens = 3;
    gp.eos = 1;
    GenResult gr;
    CHECK(ek.generate("hi", gp, gr, err) == Status::Ok);
    {
        std::ifstream uf(kdir + "/.coli_usage", std::ios::binary);
        CHECK(uf.good());
    }
    ExpertStoreStats st{};
    ek.expert_stats(st);
    CHECK(st.requests > 0);
    CHECK(st.misses + st.hits == st.requests);
    CHECK(gr.completion_tokens > 0);
    {
        RouteTelem rt;
        ek.route_telem(rt, false);
        CHECK(rt.cols == 4);
        CHECK(rt.rows >= 1);
        CHECK(static_cast<int>(rt.emap.size()) == rt.rows * rt.cols);
        CHECK(static_cast<int>(rt.hits.size()) == rt.rows * rt.cols);
        CHECK(static_cast<int>(rt.entropy.size()) == rt.rows);
        CHECK(rt.ram + rt.disk == rt.rows * rt.cols);
        int hits = 0, heat = 0;
        for (uint8_t h : rt.hits)
            hits += h ? 1 : 0;
        for (uint8_t b : rt.emap)
            if (b & 63)
                ++heat;
        CHECK(hits > 0);
        CHECK(heat > 0);
        std::string turn = mux_format_turn_telem(
            "HWINFO 1 1.00 1.00 0 0.00 cpu|none", 1, 0, 0, 0, 0, 0, 0, 0, rt.entropy.data(),
            static_cast<int>(rt.entropy.size()), rt.vram, rt.ram, rt.disk, rt.vram_gb, rt.ram_gb,
            rt.rows, rt.cols, rt.emap.data(), rt.rows, rt.cols, rt.hits.data());
        CHECK(turn.find("EMAP ") != std::string::npos);
        CHECK(turn.find("HITS ") != std::string::npos);
        CHECK(turn.find("ENTROPY ") != std::string::npos);
        CHECK(ek.hits_seq() == 0);
        std::string ej = experts_json(&ek, true);
        CHECK(ej.find("\"rows\":") != std::string::npos);
        CHECK(ej.find("\"map\":\"") != std::string::npos);
        CHECK(ej.find("\"entropy\":[") != std::string::npos);
        CHECK(ej.find("\"created\":" + std::to_string(serve_created())) != std::string::npos);
        CHECK(experts_json(&ek, false) ==
              mux_format_experts_json(0, 0, nullptr, nullptr, 0, nullptr, 0, serve_created()));
        RouteTelem a, b;
        ek.route_telem(a, true);
        CHECK(ek.hits_seq() == 1);
        ek.route_telem(b, false);
        int after = 0;
        for (uint8_t h : b.hits)
            after += h ? 1 : 0;
        CHECK(after == 0);
        CHECK(b.entropy.size() == a.entropy.size()); // consume clears hits only
        TurnPerf pf;
        ek.turn_perf(pf, false);
        CHECK(pf.t_attn > 0.0);
        CHECK(pf.t_emm > 0.0);
        CHECK(pf.t_head > 0.0);
        CHECK(pf.t_edisk >= 0.0);
        CHECK(pf.t_ewait >= 0.0);
        ek.turn_perf(pf, true);
        ek.turn_perf(pf, false);
        CHECK(pf.t_attn == 0.0 && pf.t_emm == 0.0 && pf.t_head == 0.0);
        CHECK(ek.profile_seq() >= 1);
        std::vector<ProfileTurn> pturns;
        ek.profile_turns(pturns);
        CHECK(!pturns.empty());
        CHECK(pturns.back().completion_tokens > 0);
        std::string pj = profile_json(&ek, true);
        CHECK(pj.find("\"seq\":") != std::string::npos);
        CHECK(pj.find("\"turns\":[") != std::string::npos);
        CHECK(pj.find("\"wall_s\":") != std::string::npos);
        CHECK(pj.find("\"created\":" + std::to_string(serve_created())) != std::string::npos);
        const std::string pempty =
            "{\"seq\":0,\"turns\":[],\"created\":" + std::to_string(serve_created()) + "}";
        CHECK(profile_json(&ek, false) == pempty);
        CHECK(profile_json(nullptr) == pempty);
        std::string hj = health_json(&ek);
        CHECK(hj.find("\"jobs\":") != std::string::npos);
        CHECK(hj.find("\"live\":") != std::string::npos);
        CHECK(hj.find("\"idle\":") != std::string::npos);
        CHECK(hj.find("\"hist_tokens\":") != std::string::npos);
        CHECK(hj.find("\"failed\":") != std::string::npos);
        CHECK(hj.find("\"free_slots\":") != std::string::npos);
        const std::string cempty =
            "{\"stats\":{},\"perf\":{},\"topk\":[],\"entropy\":[],\"gpus\":[],\"repin\":[],"
            "\"created\":" +
            std::to_string(serve_created()) + "}";
        CHECK(colibri_json(nullptr) == cempty);
        std::string cj = colibri_json(&ek);
        CHECK(cj.find("\"created\":" + std::to_string(serve_created())) != std::string::npos);
        CHECK(cj.find("\"stats\":{") != std::string::npos);
        CHECK(cj.find("\"profile_seq\":") != std::string::npos);
        CHECK(cj.find("\"perf\":{") != std::string::npos);
        CHECK(openai_sse_colibri(nullptr).find("data: {\"colibri\":") == 0);
        CHECK(with_colibri("{\"ok\":true}", nullptr).find("\"colibri\":") != std::string::npos);
    }

    std::string gdir = tmpdir();
    write_file(gdir + "/config.json", R"({
      "model_type": "glm",
      "architectures": ["Glm5ForConditionalGeneration"],
      "hidden_size": 32,
      "num_hidden_layers": 4,
      "vocab_size": 40,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_experts": 4,
      "num_experts_per_token": 2,
      "moe_intermediate_size": 16,
      "num_shared_experts": 1,
      "routed_scaling_factor": 2.5,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 16,
      "hc_mult": 2,
      "layer_types": ["linear","linear","linear","full_attention"],
      "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4},
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    Engine eg;
    CHECK(eg.load(gdir, rt, err) == Status::Ok);
    CHECK(eg.generate("ok", gp, gr, err) == Status::Ok);
    {
        std::ifstream uf(gdir + "/.coli_usage", std::ios::binary);
        CHECK(uf.good());
    }
    ExpertStoreStats gst{};
    eg.expert_stats(gst);
    CHECK(gst.requests > 0);
    CHECK(gst.misses + gst.hits == gst.requests);
    CHECK(gr.completion_tokens > 0);
    {
        RouteTelem rt;
        eg.route_telem(rt, true);
        CHECK(rt.cols == 4);
        CHECK(rt.rows >= 1);
        CHECK(static_cast<int>(rt.emap.size()) == rt.rows * rt.cols);
        CHECK(static_cast<int>(rt.hits.size()) == rt.rows * rt.cols);
        int hits = 0;
        for (uint8_t h : rt.hits)
            hits += h ? 1 : 0;
        CHECK(hits > 0);
        CHECK(rt.ram + rt.disk == rt.rows * rt.cols);
        RouteTelem cleared;
        eg.route_telem(cleared, false);
        int after = 0;
        for (uint8_t h : cleared.hits)
            after += h ? 1 : 0;
        CHECK(after == 0);
        TurnPerf pf;
        eg.turn_perf(pf, false);
        CHECK(pf.t_attn > 0.0);
        CHECK(pf.t_emm > 0.0);
        CHECK(pf.t_head > 0.0);
    }

    std::string hdir = tmpdir();
    write_file(hdir + "/config.json", R"({"model_type":"minimax_h3","architectures":["MiniMaxH3"]})");
    Engine eh;
    CHECK(eh.load(hdir, rt, err) == Status::Ok);
    H3GenParams hp;
    hp.prompt = "fox";
    hp.steps = 2;
    hp.dit_layers = 4;
    hp.width = 32;
    hp.height = 32;
    hp.frames = 5;
    hp.output_path = hdir + "/out.txt";
    H3GenResult hr;
    CHECK(eh.generate_video(hp, hr, err) == Status::Ok);
    CHECK(eh.block_hits() + eh.block_misses() > 0);
    CHECK(hr.blocks_streamed == 8);
    CHECK(hr.vae_used);
    CHECK(hr.audio_used);
    CHECK(hr.audio_samples > 0);
    CHECK(hr.note.find("audio=synth") != std::string::npos);
}

static void write_le_u64(std::ostream &o, uint64_t v) {
    for (int i = 0; i < 8; ++i)
        o.put(static_cast<char>((v >> (8 * i)) & 0xff));
}

static void write_safetensors_file(
    const std::string &path,
    const std::vector<std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>>
        &tensors) {
    std::string header = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto &name = std::get<0>(tensors[i]);
        const auto &dtype = std::get<1>(tensors[i]);
        const auto &shape = std::get<2>(tensors[i]);
        const auto &data = std::get<3>(tensors[i]);
        if (i)
            header += ",";
        header += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
        for (size_t s = 0; s < shape.size(); ++s) {
            if (s)
                header += ",";
            header += std::to_string(shape[s]);
        }
        header += "],\"data_offsets\":[" + std::to_string(off) + "," +
                  std::to_string(off + data.size()) + "]}";
        off += data.size();
    }
    header += "}";
    std::ofstream o(path, std::ios::binary);
    write_le_u64(o, header.size());
    o.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto &t : tensors) {
        const auto &data = std::get<3>(t);
        o.write(reinterpret_cast<const char *>(data.data()),
                static_cast<std::streamsize>(data.size()));
    }
}

static void test_glm53_container() {
    using namespace mvllm;
    std::string dir = tmpdir();
    write_file(dir + "/config.json", R"({
      "model_type": "glm",
      "architectures": ["Glm5ForConditionalGeneration"],
      "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "first_k_dense_replace": 1,
        "intermediate_size": 64,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 64,
        "n_shared_experts": 1,
        "routed_scaling_factor": 2.5,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "v_head_dim": 16,
        "hc_mult": 2,
        "layer_types": ["linear_attention", "linear_attention"],
        "linear_attn_config": {
          "num_heads": 2,
          "head_dim": 16,
          "short_conv_kernel_size": 4
        }
      },
      "bos_token_id": 0,
      "eos_token_id": 1
    })");

    ModelConfig cfg;
    std::string err;
    CHECK(load_model_config(dir, cfg, err) == Status::Ok);
    CHECK(cfg.family == Family::Glm53);
    CHECK(cfg.moe.n_experts == 2);
    CHECK(cfg.moe.topk == 1);
    CHECK(cfg.hidden == 64);
    CHECK(cfg.first_dense == 1);

    const int64_t pack = 64ll * 64 / 2;
    const int64_t scn = 64ll * 64 / 64;
    auto u8 = [&](size_t n) { return std::vector<uint8_t>(n, 0); };
    auto f32z = [&](size_t n) { return std::vector<uint8_t>(n * 4, 0); };
    auto bf16z = [&](size_t n) { return std::vector<uint8_t>(n * 2, 0); };

    using T = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
    std::vector<T> ts;
    ts.push_back({"model.embed_tokens.weight", "BF16", {32, 64}, bf16z(32 * 64)});
    ts.push_back({"model.norm.weight", "F32", {64}, f32z(64)});
    const char *pieces[6] = {"gate_proj.weight",     "gate_proj.weight.qs", "up_proj.weight",
                             "up_proj.weight.qs",    "down_proj.weight",    "down_proj.weight.qs"};
    const int64_t plen[6] = {pack, scn * 4, pack, scn * 4, pack, scn * 4};
    for (int e = 0; e < 2; ++e) {
        for (int p = 0; p < 6; ++p) {
            std::string name = "model.layers.1.mlp.experts." + std::to_string(e) + "." + pieces[p];
            bool qs = std::string(pieces[p]).find(".qs") != std::string::npos;
            if (qs)
                ts.push_back({name, "F32", {scn}, f32z(static_cast<size_t>(scn))});
            else
                ts.push_back({name, "U8", {plen[p]}, u8(static_cast<size_t>(plen[p]))});
        }
    }
    write_safetensors_file(dir + "/model.safetensors", ts);

    Engine eg;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    CHECK(eg.load(dir, rt, err) == Status::Ok);
    std::string info = eg.info();
    CHECK(info.find("checkpoint=yes") != std::string::npos);
    CHECK(info.find("bits=4") != std::string::npos);
    const bool glm_coli_tier_cpu = info.find("coli=cpu") != std::string::npos;
    CHECK(glm_coli_tier_cpu);
    const bool glm_coli_avail = mvllm::coli_cuda::available();
    CHECK(glm_coli_avail);
    const bool glm_metal_tier = info.find("metal=") != std::string::npos;
    CHECK(glm_metal_tier);
    CHECK(eg.config().moe.n_experts == 2);
    GenParams gp;
    gp.max_new_tokens = 2;
    gp.eos = 1;
    GenResult gr;
    CHECK(eg.generate("hi", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
    ExpertStoreStats st{};
    eg.expert_stats(st);
    CHECK(st.requests > 0);
}

static void test_k3_mxfp4_container() {
    using namespace mvllm;
    std::string dir = tmpdir();
    write_file(dir + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_experts": 2,
      "num_experts_per_token": 1,
      "moe_intermediate_size": 32,
      "routed_expert_hidden_size": 32,
      "num_shared_experts": 1,
      "attn_res_block_size": 2,
      "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
        "kda_layers": [1],
        "full_attn_layers": [2]
      },
      "bos_token_id": 0,
      "eos_token_id": 1
    })");

    ModelConfig cfg;
    std::string err;
    CHECK(load_model_config(dir, cfg, err) == Status::Ok);
    CHECK(cfg.family == Family::KimiK3);
    CHECK(cfg.moe.latent == 32);
    CHECK(cfg.moe.intermediate == 32);

    const int64_t w1p = 32ll * (32 / 2);
    const int64_t w1s = 32ll * (32 / 32);
    const int64_t w2p = 32ll * (32 / 2);
    const int64_t w2s = 32ll * (32 / 32);
    auto u8 = [&](size_t n) { return std::vector<uint8_t>(n, 0); };
    auto bf16z = [&](size_t n) { return std::vector<uint8_t>(n * 2, 0); };

    using T = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
    std::vector<T> ts;
    ts.push_back({"model.embed_tokens.weight", "BF16", {32, 32}, bf16z(32 * 32)});
    ts.push_back({"model.layers.0.input_layernorm.weight", "BF16", {32}, bf16z(32)});
    const char *mats[3] = {"w1", "w2", "w3"};
    const char *half[2] = {"packed", "scale"};
    const int64_t want[6] = {w1p, w1s, w2p, w2s, w1p, w1s};
    for (int e = 0; e < 2; ++e) {
        for (int k = 0; k < 6; ++k) {
            std::string name = "model.layers.1.block_sparse_moe.experts." + std::to_string(e) +
                               "." + mats[k / 2] + ".weight_" + half[k & 1];
            ts.push_back({name, "U8", {want[k]}, u8(static_cast<size_t>(want[k]))});
        }
    }
    write_safetensors_file(dir + "/model.safetensors", ts);

    Engine ek;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    CHECK(ek.load(dir, rt, err) == Status::Ok);
    std::string info = ek.info();
    CHECK(info.find("checkpoint=yes") != std::string::npos);
    CHECK(info.find("bits=4") != std::string::npos);
    const bool k3_coli_tier_cpu = info.find("coli=cpu") != std::string::npos;
    CHECK(k3_coli_tier_cpu);
    const bool k3_coli_avail = mvllm::coli_cuda::available();
    CHECK(k3_coli_avail);
    const bool k3_coli_ndev = mvllm::coli_cuda::available_device_count() == 0;
    CHECK(k3_coli_ndev);
    const bool k3_metal_tier = info.find("metal=") != std::string::npos;
    CHECK(k3_metal_tier);
    GenParams gp;
    gp.max_new_tokens = 2;
    gp.eos = 1;
    GenResult gr;
    CHECK(ek.generate("hi", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
    ExpertStoreStats st{};
    ek.expert_stats(st);
    CHECK(st.requests > 0);
}

static void test_dense_bits_generate() {
    using namespace mvllm;
    std::string err;
    const char *kcfg = R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "first_k_dense_replace": 1,
      "intermediate_size": 64,
      "num_experts": 2,
      "num_experts_per_token": 1,
      "moe_intermediate_size": 32,
      "routed_expert_hidden_size": 32,
      "num_shared_experts": 1,
      "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 16,
        "short_conv_kernel_size": 4,
        "kda_layers": [1],
        "full_attn_layers": [2]
      },
      "bos_token_id": 0,
      "eos_token_id": 1
    })";
    const char *gcfg = R"({
      "model_type": "glm",
      "architectures": ["Glm5ForConditionalGeneration"],
      "hidden_size": 64,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "first_k_dense_replace": 1,
      "intermediate_size": 64,
      "n_routed_experts": 2,
      "num_experts_per_tok": 1,
      "moe_intermediate_size": 32,
      "n_shared_experts": 1,
      "routed_scaling_factor": 2.5,
      "linear_attn_config": {"num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 4},
      "layer_types": ["linear_attention", "linear_attention"],
      "bos_token_id": 0,
      "eos_token_id": 1
    })";

    auto run = [&](const char *cfg, int bits) {
        std::string dir = tmpdir();
        write_file(dir + "/config.json", cfg);
        Engine e;
        RuntimeConfig rt;
        rt.expert_gb = 0.01;
        rt.dense_bits = bits;
        rt.head_bits = bits == 32 ? 32 : 8;
        CHECK(e.load(dir, rt, err) == Status::Ok);
        std::string info = e.info();
        CHECK(info.find("bits=" + std::to_string(bits)) != std::string::npos);
        GenParams gp;
        gp.max_new_tokens = 2;
        gp.eos = 1;
        GenResult gr;
        CHECK(e.generate("hi", gp, gr, err) == Status::Ok);
        CHECK(gr.completion_tokens > 0);
    };
    run(kcfg, 4);
    run(kcfg, 8);
    run(kcfg, 32);
    run(gcfg, 4);
    run(gcfg, 8);
    run(gcfg, 32);
}

static void test_h3_checkpoint() {
    using namespace mvllm;
    std::string dir = tmpdir();
    write_file(dir + "/config.json", R"({"model_type":"minimax_h3","architectures":["MiniMaxH3"]})");

    const int hidden = 8, inner = 4, ffn = 8;
    auto bf16z = [&](size_t n) { return std::vector<uint8_t>(n * 2, 0); };
    using T = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
    std::vector<T> ts;
    for (int b = 0; b < 2; ++b) {
        std::string p = "blocks." + std::to_string(b) + ".";
        ts.push_back({p + "attn.qkv_proj.weight", "BF16", {inner * 3, hidden},
                      bf16z(static_cast<size_t>(inner * 3 * hidden))});
        ts.push_back({p + "attn.out_proj.weight", "BF16", {hidden, inner},
                      bf16z(static_cast<size_t>(hidden * inner))});
        ts.push_back({p + "mlp.fc1.weight", "BF16", {ffn * 2, hidden},
                      bf16z(static_cast<size_t>(ffn * 2 * hidden))});
        ts.push_back({p + "mlp.fc2.weight", "BF16", {hidden, ffn},
                      bf16z(static_cast<size_t>(hidden * ffn))});
    }
    write_safetensors_file(dir + "/model.safetensors", ts);

    Engine eh;
    RuntimeConfig rt;
    std::string err;
    CHECK(eh.load(dir, rt, err) == Status::Ok);
    CHECK(eh.config().h3.dit_layers == 2);
    CHECK(eh.config().h3.hidden == hidden);
    CHECK(eh.config().h3.inner == inner);
    CHECK(eh.config().h3.ffn == ffn);
    std::string info = eh.info();
    CHECK(info.find("checkpoint=yes") != std::string::npos);
    H3GenParams hp;
    hp.prompt = "fox";
    hp.steps = 2;
    hp.dit_layers = 2;
    hp.width = 32;
    hp.height = 32;
    hp.frames = 5;
    hp.output_path = dir + "/out.txt";
    H3GenResult hr;
    CHECK(eh.generate_video(hp, hr, err) == Status::Ok);
    CHECK(hr.blocks_streamed == 4);
    CHECK(eh.block_hits() + eh.block_misses() > 0);
    CHECK(hr.note.find("CPU DiT") != std::string::npos);
    CHECK(hr.note.find("latent_tokens=") != std::string::npos);
    CHECK(hr.note.find("VAE latent") != std::string::npos);
    CHECK(hr.blocks_streamed == 4);
    CHECK(hr.vae_used);
    CHECK(hr.frames == 5);
    CHECK(hr.width == 32 && hr.height == 32);
}

static void test_shard_probe() {
    using namespace mvllm;
    CHECK(k3_expert_slot_bytes(32, 32) == 2 * (32 * 16 + 32 * 1) + 32 * 16 + 32 * 1);
    CHECK(glm53_expert_slot_bytes(64, 64) == 3 * (64 * 64 / 2 + 64 * 64 / 64 * 4));

    std::string dir = tmpdir();
    write_file(dir + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "num_experts": 2,
      "moe_intermediate_size": 32,
      "routed_expert_hidden_size": 32
    })");
    const int64_t w1p = 32ll * 16, w1s = 32ll;
    auto u8 = [&](size_t n) { return std::vector<uint8_t>(n, 0); };
    using T = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
    std::vector<T> ts;
    const char *mats[3] = {"w1", "w2", "w3"};
    const char *half[2] = {"packed", "scale"};
    const int64_t want[6] = {w1p, w1s, w1p, w1s, w1p, w1s};
    for (int e = 0; e < 2; ++e) {
        for (int k = 0; k < 6; ++k) {
            std::string name = "language_model.model.layers.1.block_sparse_moe.experts." +
                               std::to_string(e) + "." + mats[k / 2] + ".weight_" + half[k & 1];
            ts.push_back({name, "U8", {want[k]}, u8(static_cast<size_t>(want[k]))});
        }
    }
    write_safetensors_file(dir + "/model-00001-of-00001.safetensors", ts);

    ShardReport r;
    RuntimeConfig rt;
    rt.expert_gb = 8;
    std::string err;
    CHECK(probe_shards(dir, rt, r, err) == Status::Ok);
    CHECK(r.family == Family::KimiK3);
    CHECK(r.prefix == "language_model.");
    CHECK(r.prefix_ok);
    CHECK(r.n_experts_seen == 2);
    CHECK(r.n_layers_seen == 2);
    CHECK(r.slot_bytes == k3_expert_slot_bytes(32, 32));
    CHECK(r.slots_per_layer <= 2);
    CHECK(r.shards_ok);
    CHECK(r.payload_ok);
    CHECK(r.payload_bytes > 0);
    CHECK(!r.synth);
    std::string text = format_shard_report(r);
    CHECK(text.find("prefix=language_model.") != std::string::npos);
    CHECK(text.find("payload_ok=yes") != std::string::npos);

    std::string mismatch = tmpdir();
    write_file(mismatch + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "num_experts": 4,
      "moe_intermediate_size": 32,
      "routed_expert_hidden_size": 32
    })");
    write_safetensors_file(mismatch + "/model-00001-of-00001.safetensors", ts);
    ShardReport rm;
    CHECK(probe_shards(mismatch, rt, rm, err) == Status::Ok);
    CHECK(!rm.shards_ok);
    CHECK(rm.note.find("counted experts") != std::string::npos);

    std::string empty = tmpdir();
    write_file(empty + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "num_experts": 2
    })");
    ShardReport re;
    CHECK(probe_shards(empty, rt, re, err) == Status::Ok);
    CHECK(re.synth);
    CHECK(re.shards_ok);

    std::string hdir = tmpdir();
    write_file(hdir + "/config.json", R"({"model_type":"minimax_h3","architectures":["MiniMaxH3"]})");
    using HT = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
    std::vector<HT> hts;
    auto bf16z = [&](size_t n) { return std::vector<uint8_t>(n * 2, 0); };
    hts.push_back({"blocks.0.attn.qkv_proj.weight", "BF16", {12, 8}, bf16z(96)});
    hts.push_back({"rope.inv_freq", "F32", {16}, std::vector<uint8_t>(16 * 4, 0)});
    write_safetensors_file(hdir + "/model.safetensors", hts);
    ShardReport rh;
    CHECK(probe_shards(hdir, rt, rh, err) == Status::Ok);
    CHECK(rh.family == Family::H3);
    CHECK(rh.prefix_ok);
    CHECK(rh.payload_ok);
    CHECK(rh.shards_ok);
    CHECK(format_shard_report(rh).find("transformer") != std::string::npos);

    std::string bad = tmpdir();
    write_file(bad + "/config.json", R"({"model_type":"kimi_linear"})");
    write_file(bad + "/model.safetensors", std::string(8, '\0'));
    ShardReport rb;
    std::string berr;
    CHECK(probe_shards(bad, rt, rb, berr) != Status::Ok);
}

static bool dump_env_readable_dir(const char *p) {
    if (!p || !p[0])
        return false;
    struct stat st {};
    if (::stat(p, &st) != 0 || !S_ISDIR(st.st_mode))
        return false;
    return ::access(p, R_OK) == 0;
}

static void test_live_dump_env() {
    using namespace mvllm;
    CHECK(dump_env_for(Family::KimiK3) == dump_env_k3());
    CHECK(dump_env_for(Family::Glm53) == dump_env_glm53());
    CHECK(dump_env_for(Family::H3) == dump_env_h3());
    CHECK(dump_env_for(Family::Dsv4) == dump_env_dsv4());
    CHECK(dump_env_for(Family::Llama) == nullptr);
    CHECK(dump_env_for(Family::Unknown) == nullptr);
    const char *dsv4 = dump_env_dsv4();
    CHECK(dsv4 == nullptr || dsv4[0] != '\0');

    RuntimeConfig rt;
    rt.expert_gb = 8;

    if (const char *k3 = dump_env_k3(); dump_env_readable_dir(k3)) {
        const bool live_dump_k3_sniff = sniff_family(k3) == Family::KimiK3;
        CHECK(live_dump_k3_sniff);
        ShardReport live_k3_rep;
        std::string live_k3_err;
        const bool live_dump_k3_probe =
            probe_shards(k3, rt, live_k3_rep, live_k3_err) == Status::Ok;
        CHECK(live_dump_k3_probe);
        const bool live_dump_k3_family = live_k3_rep.family == Family::KimiK3;
        CHECK(live_dump_k3_family);
    }
    if (const char *glm = dump_env_glm53(); dump_env_readable_dir(glm)) {
        const bool live_dump_glm53_sniff = sniff_family(glm) == Family::Glm53;
        CHECK(live_dump_glm53_sniff);
        ShardReport live_glm53_rep;
        std::string live_glm53_err;
        const bool live_dump_glm53_probe =
            probe_shards(glm, rt, live_glm53_rep, live_glm53_err) == Status::Ok;
        CHECK(live_dump_glm53_probe);
        const bool live_dump_glm53_family = live_glm53_rep.family == Family::Glm53;
        CHECK(live_dump_glm53_family);
    }
    if (const char *h3 = dump_env_h3(); dump_env_readable_dir(h3)) {
        const bool live_dump_h3_sniff = sniff_family(h3) == Family::H3;
        CHECK(live_dump_h3_sniff);
        ShardReport live_h3_rep;
        std::string live_h3_err;
        const bool live_dump_h3_probe =
            probe_shards(h3, rt, live_h3_rep, live_h3_err) == Status::Ok;
        CHECK(live_dump_h3_probe);
        const bool live_dump_h3_family = live_h3_rep.family == Family::H3;
        CHECK(live_dump_h3_family);
    }
    if (const char *v4 = dump_env_dsv4(); dump_env_readable_dir(v4)) {
        const bool live_dump_dsv4_sniff = sniff_family(v4) == Family::Dsv4;
        CHECK(live_dump_dsv4_sniff);
        ShardReport live_dsv4_rep;
        std::string live_dsv4_err;
        const bool live_dump_dsv4_probe =
            probe_shards(v4, rt, live_dsv4_rep, live_dsv4_err) == Status::Ok;
        CHECK(live_dump_dsv4_probe);
        const bool live_dump_dsv4_family = live_dsv4_rep.family == Family::Dsv4;
        CHECK(live_dump_dsv4_family);
    }
}

static void test_dsa() {
    using namespace mvllm;
    DsaConfig d;
    d.topk = 4;
    d.kpool = 2;
    d.always_select_tail = true;
    CHECK(dsa_index_width(d) == 5);
    d.always_select_tail = false;
    CHECK(dsa_index_width(d) == 4);

    d.n_heads = 1;
    d.head_dim = 2;
    d.topk = 2;
    d.kpool = 2;
    d.always_select_tail = true;
    float q[2] = {1.f, 0.f};
    float keys[10] = {1.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.5f, 0.5f};
    float gates[10] = {};
    float hw[1] = {1.f};
    int sel[8];
    int w = dsa_select(sel, q, keys, gates, hw, nullptr, 5, d);
    CHECK(w == 3);
    CHECK(sel[2] == 4); // tail token
    bool has0 = sel[0] == 0 || sel[1] == 0;
    CHECK(has0);

    MlaConfig mla;
    mla.n_heads = 1;
    mla.q_lora = 4;
    mla.kv_lora = 4;
    mla.qk_nope = 4;
    mla.v_head = 4;
    mla.output_gate = false;
    quant::QuantMat qa, qb, kva, kt, vv, wo;
    std::vector<float> qa_w(4 * 8), qb_w(4 * 4, 0.2f), kva_w(4 * 8, 0.15f), wo_w(8 * 4, 0.05f);
    for (int i = 0; i < 32; ++i)
        qa_w[i] = ((i % 5) - 2) * 0.1f;
    qa.from_f32(qa_w.data(), 4, 8, 32);
    qb.from_f32(qb_w.data(), 4, 4, 32);
    kva.from_f32(kva_w.data(), 4, 8, 32);
    std::vector<float> kvb(1 * (4 + 4) * 4, 0.08f);
    mla_absorb_kvb(kvb.data(), 1, 4, 4, 4, kt, vv, 32);
    wo.from_f32(wo_w.data(), 8, 4, 32);
    std::vector<float> ln(4, 1.f), cache(8 * 4, 0.f), x(8, 0.2f), y0(8), y1(8);
    int one[1] = {0};
    mla_step(x.data(), 8, mla, &qa, ln.data(), &qb, &kva, ln.data(), &kt, &vv, &wo, nullptr,
             cache.data(), 1, y0.data(), 1e-5f, nullptr, 0);
    mla_step(x.data(), 8, mla, &qa, ln.data(), &qb, &kva, ln.data(), &kt, &vv, &wo, nullptr,
             cache.data(), 1, y1.data(), 1e-5f, one, 1);
    float diff = 0.f;
    for (int i = 0; i < 8; ++i) {
        CHECK(std::isfinite(y0[i]) && std::isfinite(y1[i]));
        diff += (y0[i] - y1[i]) * (y0[i] - y1[i]);
    }
    CHECK(diff > 0.f);
}

static void test_moe_union() {
    using namespace mvllm;
    float sig[4] = {0.1f, 0.4f, 0.3f, 0.2f};
    float choice[4] = {0.1f, 0.4f, 10.3f, 0.2f}; // expert 2 wins selection via bias
    int top[2];
    float wt[2];
    CHECK(moe_topk(choice, 4, 2, top, wt, sig) == 2);
    CHECK(top[0] == 2);
    CHECK_NEAR(wt[0], 0.3f / (0.3f + 0.4f), 1e-5); // mix uses unbiased σ, not 10.3
    CHECK(top[1] == 1);
    CHECK_NEAR(wt[1], 0.4f / (0.3f + 0.4f), 1e-5);

    int idx[] = {3, 1, 3, 2, 1, -1};
    int out[8];
    int n = moe_union_ids(idx, 3, 2, out, 8);
    CHECK(n == 3);
    CHECK(out[0] == 1 && out[1] == 2 && out[2] == 3);

    std::string err;
    std::string kdir = tmpdir();
    write_file(kdir + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_experts": 2,
      "num_experts_per_token": 1,
      "moe_intermediate_size": 16,
      "routed_expert_hidden_size": 16,
      "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4,
        "kda_layers": [1], "full_attn_layers": [2]},
      "bos_token_id": 0, "eos_token_id": 1
    })");
    Engine ek;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    rt.prefill_chunk = 2;
    CHECK(ek.load(kdir, rt, err) == Status::Ok);
    CHECK(ek.info().find("prefill=layer") != std::string::npos);
    GenParams gp;
    gp.max_new_tokens = 2;
    gp.eos = 1;
    gp.apply_template = false;
    GenResult gr;
    CHECK(ek.generate("hi!!", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
}

static void test_kda_short_conv() {
    using namespace mvllm;
    const int P = 2, K = 4;
    float x0[2] = {1.f, 2.f};
    float x1[2] = {3.f, 4.f};
    float taps[8];
    for (int i = 0; i < 8; ++i)
        taps[i] = (i % K == K - 1) ? 1.f : 0.f; // last tap = 1 → identity after fill
    float win[8] = {};
    kda_short_conv(x0, taps, win, P, K);
    // first step: window was zeros, last = x, tap last=1 → out = x
    CHECK_NEAR(x0[0], 1.f, 1e-5);
    CHECK_NEAR(x0[1], 2.f, 1e-5);
    kda_short_conv(x1, taps, win, P, K);
    CHECK_NEAR(x1[0], 3.f, 1e-5);
    CHECK_NEAR(x1[1], 4.f, 1e-5);

    float all_one_taps[8];
    for (int i = 0; i < 8; ++i)
        all_one_taps[i] = 1.f;
    float win2[8] = {};
    float a[2] = {1.f, 1.f};
    float b[2] = {1.f, 1.f};
    kda_short_conv(a, all_one_taps, win2, P, K);
    kda_short_conv(b, all_one_taps, win2, P, K);
    // after 2 steps window is [0,0,1,1] · [1,1,1,1] = 2
    CHECK_NEAR(b[0], 2.f, 1e-5);

    // Shared [head_dim] o_norm must not be indexed as [heads, head_dim].
    KdaConfig kda;
    kda.heads = 2;
    kda.head_dim = 4;
    std::vector<float> xin(8, 0.1f), S(2 * 4 * 4, 0.f), y(8, 0.f), on(4, 2.f);
    kda_step(xin.data(), 8, kda, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
             nullptr, nullptr, nullptr, nullptr, on.data(), S.data(), y.data(), 1e-6f);
    for (float v : y)
        CHECK(std::isfinite(v));

    // mHC of [x, 0] is not a clone of x.
    float st[4] = {1.f, 2.f, 0.f, 0.f};
    mhc_mix(st, 2, 2, nullptr, 20, 1e-6f);
    CHECK(std::fabs(st[0] - 1.f) > 1e-4f);
    CHECK(std::isfinite(st[0]) && std::isfinite(st[2]));

    // Official mHC pre/post: identity-ish collapse of stream 0.
    const int M = 2, D = 2;
    std::vector<float> streams = {1.f, 2.f, 3.f, 4.f};
    std::vector<float> fn((2 + M) * M * M * D, 0.f);
    std::vector<float> base((2 + M) * M, 0.f);
    std::vector<float> scale = {0.f, 0.f, 0.f};
    // mixes ≈ 0 → pre ≈ σ(0)+eps, post ≈ 2σ(0), comb row-softmax of zeros = uniform
    std::vector<float> collapsed(D), post(M), comb(M * M);
    CHECK(mhc_pre(collapsed.data(), post.data(), comb.data(), streams.data(), fn.data(),
                  scale.data(), base.data(), M, D, 20, 1e-6f, 1e-6f) == 0);
    CHECK(std::isfinite(collapsed[0]) && std::isfinite(collapsed[1]));
    std::vector<float> branch = {0.5f, -0.25f};
    std::vector<float> outst(M * D, 0.f);
    CHECK(mhc_post(outst.data(), branch.data(), streams.data(), post.data(), comb.data(), M, D) ==
          0);
    for (float v : outst)
        CHECK(std::isfinite(v));
}

static void test_mla_absorb() {
    using namespace mvllm;
    const int H = 2, QK = 4, Vh = 4, L = 8, hidden = 16;
    std::vector<float> kv_b(static_cast<size_t>(H) * (QK + Vh) * L);
    for (size_t i = 0; i < kv_b.size(); ++i)
        kv_b[i] = ((static_cast<int>(i) * 13) % 11 - 5) * 0.05f;
    quant::QuantMat kt, vv;
    mla_absorb_kvb(kv_b.data(), H, QK, Vh, L, kt, vv, 32);
    CHECK(kt.fmt == 0 && kt.O == H * L && kt.I == QK);
    CHECK(vv.fmt == 0 && vv.O == H * Vh && vv.I == L);

    // Absorb is a transpose of W_k and a copy of W_v.
    for (int h = 0; h < H; ++h) {
        const float *block = kv_b.data() + static_cast<size_t>(h) * (QK + Vh) * L;
        for (int d = 0; d < L; ++d)
            for (int i = 0; i < QK; ++i)
                CHECK_NEAR(kt.f[(static_cast<size_t>(h) * L + d) * QK + i],
                           block[static_cast<size_t>(i) * L + d], 1e-6);
        for (int i = 0; i < Vh * L; ++i)
            CHECK_NEAR(vv.f[static_cast<size_t>(h) * Vh * L + i],
                       block[static_cast<size_t>(QK) * L + i], 1e-6);
    }

    MlaConfig mla;
    mla.n_heads = H;
    mla.q_lora = 8;
    mla.kv_lora = L;
    mla.qk_nope = QK;
    mla.v_head = Vh;
    mla.output_gate = false;

    quant::QuantMat qa, qb, kva, wo;
    std::vector<float> qa_w(static_cast<size_t>(mla.q_lora) * hidden, 0.1f);
    std::vector<float> qb_w(static_cast<size_t>(H) * QK * mla.q_lora, 0.05f);
    std::vector<float> kva_w(static_cast<size_t>(L) * hidden, 0.08f);
    std::vector<float> wo_w(static_cast<size_t>(hidden) * H * Vh, 0.04f);
    for (size_t i = 0; i < qa_w.size(); ++i)
        qa_w[i] = ((static_cast<int>(i) % 7) - 3) * 0.04f;
    for (size_t i = 0; i < qb_w.size(); ++i)
        qb_w[i] = ((static_cast<int>(i) * 3) % 9 - 4) * 0.06f;
    for (size_t i = 0; i < kva_w.size(); ++i)
        kva_w[i] = ((static_cast<int>(i) * 5) % 11 - 5) * 0.05f;
    for (size_t i = 0; i < wo_w.size(); ++i)
        wo_w[i] = ((static_cast<int>(i) * 7) % 13 - 6) * 0.07f;
    qa.from_f32(qa_w.data(), mla.q_lora, hidden, 32);
    qb.from_f32(qb_w.data(), H * QK, mla.q_lora, 32);
    kva.from_f32(kva_w.data(), L, hidden, 32);
    wo.from_f32(wo_w.data(), hidden, H * Vh, 32);
    std::vector<float> qa_ln(mla.q_lora, 1.f), kva_ln(L, 1.f);
    std::vector<float> cache(static_cast<size_t>(4) * L, 0.f);
    std::vector<float> x(hidden, 0.2f), y0(hidden), y1(hidden);
    for (int i = 0; i < hidden; ++i)
        x[i] = ((i % 5) - 2) * 0.1f;
    mla_step(x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &kt, &vv, &wo,
             nullptr, cache.data(), 0, y0.data(), 1e-5f);
    for (int i = 0; i < hidden; ++i)
        x[i] = ((i % 7) - 3) * 0.25f;
    mla_step(x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &kt, &vv, &wo,
             nullptr, cache.data(), 1, y1.data(), 1e-5f);
    float n0 = 0.f, n1 = 0.f, diff = 0.f;
    for (int i = 0; i < hidden; ++i) {
        n0 += y0[i] * y0[i];
        n1 += y1[i] * y1[i];
        diff += (y0[i] - y1[i]) * (y0[i] - y1[i]);
        CHECK(std::isfinite(y0[i]) && std::isfinite(y1[i]));
    }
    CHECK(n0 > 0.f && n1 > 0.f);
    CHECK(diff > 1e-8f);

    quant::QuantMat mla_q8_kt, mla_q8_vv;
    mla_absorb_kvb(kv_b.data(), H, QK, Vh, L, mla_q8_kt, mla_q8_vv, 8);
    std::vector<float> mla_q8_cache(static_cast<size_t>(4) * L, 0.f);
    std::vector<float> mla_q8_x(hidden), mla_q8_y0(hidden), mla_q8_y1(hidden);
    for (int i = 0; i < hidden; ++i)
        mla_q8_x[i] = ((i % 5) - 2) * 0.1f;
    mla_step(mla_q8_x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &mla_q8_kt,
             &mla_q8_vv, &wo, nullptr, mla_q8_cache.data(), 0, mla_q8_y0.data(), 1e-5f);
    for (int i = 0; i < hidden; ++i)
        mla_q8_x[i] = ((i % 7) - 3) * 0.25f;
    mla_step(mla_q8_x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &mla_q8_kt,
             &mla_q8_vv, &wo, nullptr, mla_q8_cache.data(), 1, mla_q8_y1.data(), 1e-5f);
    float mla_q8_e0 = 0.f, mla_q8_e1 = 0.f;
    for (int i = 0; i < hidden; ++i) {
        CHECK(std::isfinite(mla_q8_y0[i]) && std::isfinite(mla_q8_y1[i]));
        mla_q8_e0 += mla_q8_y0[i] * mla_q8_y0[i];
        mla_q8_e1 += mla_q8_y1[i] * mla_q8_y1[i];
    }
    CHECK(mla_q8_e0 > 0.f && mla_q8_e1 > 0.f);

    quant::QuantMat mla_q4_kt, mla_q4_vv;
    mla_absorb_kvb(kv_b.data(), H, QK, Vh, L, mla_q4_kt, mla_q4_vv, 4);
    std::vector<float> mla_q4_cache(static_cast<size_t>(4) * L, 0.f);
    std::vector<float> mla_q4_x(hidden), mla_q4_y0(hidden), mla_q4_y1(hidden);
    for (int i = 0; i < hidden; ++i)
        mla_q4_x[i] = ((i % 5) - 2) * 0.1f;
    mla_step(mla_q4_x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &mla_q4_kt,
             &mla_q4_vv, &wo, nullptr, mla_q4_cache.data(), 0, mla_q4_y0.data(), 1e-5f);
    for (int i = 0; i < hidden; ++i)
        mla_q4_x[i] = ((i % 7) - 3) * 0.25f;
    mla_step(mla_q4_x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &mla_q4_kt,
             &mla_q4_vv, &wo, nullptr, mla_q4_cache.data(), 1, mla_q4_y1.data(), 1e-5f);
    float mla_q4_e0 = 0.f, mla_q4_e1 = 0.f;
    for (int i = 0; i < hidden; ++i) {
        CHECK(std::isfinite(mla_q4_y0[i]) && std::isfinite(mla_q4_y1[i]));
        mla_q4_e0 += mla_q4_y0[i] * mla_q4_y0[i];
        mla_q4_e1 += mla_q4_y1[i] * mla_q4_y1[i];
    }
    CHECK(mla_q4_e0 > 0.f && mla_q4_e1 > 0.f);

    // K3 rope strip: cache stride is kv_lora+qk_rope; R is not absorbed.
    MlaConfig mlar = mla;
    mlar.qk_rope = 4;
    quant::QuantMat qbr, kvar;
    std::vector<float> qbr_w(static_cast<size_t>(H) * (QK + 4) * mlar.q_lora);
    std::vector<float> kvar_w(static_cast<size_t>(L + 4) * hidden);
    for (size_t i = 0; i < qbr_w.size(); ++i)
        qbr_w[i] = ((static_cast<int>(i) * 3) % 9 - 4) * 0.06f;
    for (size_t i = 0; i < kvar_w.size(); ++i)
        kvar_w[i] = ((static_cast<int>(i) * 5) % 11 - 5) * 0.05f;
    qbr.from_f32(qbr_w.data(), H * (QK + 4), mlar.q_lora, 32);
    kvar.from_f32(kvar_w.data(), L + 4, hidden, 32);
    std::vector<float> cacher(static_cast<size_t>(4) * (L + 4), 0.f);
    std::vector<float> yr(hidden);
    mla_step(x.data(), hidden, mlar, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, cacher.data(), 0, yr.data(), 1e-5f);
    float nr = 0.f;
    for (int i = 0; i < hidden; ++i) {
        CHECK(std::isfinite(yr[i]));
        nr += yr[i] * yr[i];
    }
    CHECK(nr > 0.f);
    mlar.rope_theta = 10000.f;
    mlar.nope = false;
    MlaConfig mlar0 = mlar;
    mlar0.rope_theta = 0.f;
    std::vector<float> yr0(hidden), yr1(hidden), ytmp(hidden);
    std::vector<float> c0(cacher.size(), 0.f), c1(cacher.size(), 0.f);
    mla_step(x.data(), hidden, mlar0, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, c0.data(), 0, ytmp.data(), 1e-5f);
    mla_step(x.data(), hidden, mlar, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, c1.data(), 0, ytmp.data(), 1e-5f);
    for (int i = 0; i < hidden; ++i)
        x[i] = ((i % 5) - 2) * 0.15f;
    mla_step(x.data(), hidden, mlar0, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, c0.data(), 1, yr0.data(), 1e-5f);
    mla_step(x.data(), hidden, mlar, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, c1.data(), 1, yr1.data(), 1e-5f);
    float rd = 0.f;
    for (int i = 0; i < hidden; ++i) {
        CHECK(std::isfinite(yr0[i]) && std::isfinite(yr1[i]));
        rd += (yr0[i] - yr1[i]) * (yr0[i] - yr1[i]);
    }
    CHECK(rd > 1e-8f);
    MlaConfig mlar_nope = mlar;
    mlar_nope.nope = true;
    std::vector<float> yn(hidden), cn(cacher.size(), 0.f), cz(cacher.size(), 0.f);
    mla_step(x.data(), hidden, mlar_nope, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv,
             &wo, nullptr, cn.data(), 0, ytmp.data(), 1e-5f);
    mla_step(x.data(), hidden, mlar0, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, cz.data(), 0, ytmp.data(), 1e-5f);
    mla_step(x.data(), hidden, mlar_nope, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv,
             &wo, nullptr, cn.data(), 1, yn.data(), 1e-5f);
    mla_step(x.data(), hidden, mlar0, &qa, qa_ln.data(), &qbr, &kvar, kva_ln.data(), &kt, &vv, &wo,
             nullptr, cz.data(), 1, yr0.data(), 1e-5f);
    float nd = 0.f;
    for (int i = 0; i < hidden; ++i)
        nd += (yn[i] - yr0[i]) * (yn[i] - yr0[i]);
    CHECK(nd < 1e-10f);

    quant::QuantMat kt8, vv8;
    mla_absorb_kvb(kv_b.data(), H, QK, Vh, L, kt8, vv8, 8);
    CHECK(kt8.fmt == 8);
    std::vector<float> y8(hidden);
    mla_step(x.data(), hidden, mla, &qa, qa_ln.data(), &qb, &kva, kva_ln.data(), &kt8, &vv8, &wo,
             nullptr, cache.data(), 1, y8.data(), 1e-5f);
    for (int i = 0; i < hidden; ++i)
        CHECK(std::isfinite(y8[i]));
}

static void test_mla_generate() {
    using namespace mvllm;
    std::string err;
    std::string kdir = tmpdir();
    write_file(kdir + "/config.json", R"({
      "model_type": "kimi_linear",
      "architectures": ["KimiLinearForCausalLM"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "num_attention_heads": 2,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 8,
      "qk_rope_head_dim": 4,
      "rope_theta": 10000,
      "v_head_dim": 8,
      "num_experts": 2,
      "num_experts_per_token": 1,
      "moe_intermediate_size": 16,
      "routed_expert_hidden_size": 16,
      "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "kda_layers": [1],
        "full_attn_layers": [2]
      },
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    Engine ek;
    RuntimeConfig rt;
    rt.expert_gb = 0.01;
    rt.mla_bits = 8;
    CHECK(ek.load(kdir, rt, err) == Status::Ok);
    std::string info = ek.info();
    CHECK(info.find("mla=") != std::string::npos);
    GenParams gp;
    gp.max_new_tokens = 2;
    gp.eos = 1;
    GenResult gr;
    CHECK(ek.generate("hi", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);

    std::string gdir = tmpdir();
    write_file(gdir + "/config.json", R"({
      "model_type": "glm",
      "architectures": ["Glm5ForConditionalGeneration"],
      "hidden_size": 32,
      "num_hidden_layers": 2,
      "vocab_size": 32,
      "first_k_dense_replace": 1,
      "intermediate_size": 32,
      "n_routed_experts": 2,
      "num_experts_per_tok": 1,
      "moe_intermediate_size": 16,
      "num_attention_heads": 2,
      "q_lora_rank": 16,
      "kv_lora_rank": 16,
      "qk_nope_head_dim": 8,
      "v_head_dim": 8,
      "layer_types": ["linear_attention", "full_attention"],
      "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4},
      "bos_token_id": 0,
      "eos_token_id": 1
    })");
    Engine eg;
    CHECK(eg.load(gdir, rt, err) == Status::Ok);
    CHECK(eg.info().find("mla=") != std::string::npos);
    CHECK(eg.generate("ok", gp, gr, err) == Status::Ok);
    CHECK(gr.completion_tokens > 0);
}

static void test_h3_vae() {
    using namespace mvllm;
    CHECK(h3_align_frames(1) == 5);
    CHECK(h3_align_frames(5) == 5);
    CHECK(h3_align_frames(6) == 22);
    CHECK(h3_align_frames(56) == 56);
    CHECK(h3_video_latent_t(5) == 2);
    CHECK(h3_video_latent_t(22) == 7);
    CHECK(h3_video_latent_t(56) == 17);
    CHECK(h3_encoder_latent_t(56) == 14);
    int lw = 0, lh = 0;
    h3_latent_canvas(864, 480, 16, &lw, &lh);
    CHECK(lw == 54 && lh == 30);

    H3VaeGeom g = h3_vae_geom(32, 32, 5, 16, 24);
    CHECK(g.frames == 5 && g.width == 32 && g.height == 32);
    CHECK(g.latent_t == 2 && g.latent_h == 2 && g.latent_w == 2);
    CHECK(g.latent_ch == 24);

    H3Vae vae;
    std::string err;
    H3Config h3;
    CHECK(vae.load("/tmp/does-not-exist-mvllm-vae", h3, err) == Status::Ok);
    CHECK(!vae.from_checkpoint);
    CHECK(!vae.official_decode);
    CHECK(!vae.official_encode);

    const int F = g.frames, Ht = g.height, Wt = g.width;
    std::vector<float> rgb(static_cast<size_t>(F) * Ht * Wt * 3);
    for (size_t i = 0; i < rgb.size(); ++i)
        rgb[i] = (i % 17) / 16.f;
    std::vector<float> z(static_cast<size_t>(g.latent_ch) * g.latent_t * g.latent_h * g.latent_w);
    vae.encode(rgb.data(), g, z.data());
    float z2 = 0.f;
    bool zfin = true;
    for (float v : z) {
        zfin = zfin && std::isfinite(v);
        z2 += v * v;
    }
    CHECK(zfin);
    CHECK(z2 > 0.f);

    std::vector<float> out(rgb.size(), 0.f);
    vae.decode(z.data(), g, out.data());
    float o2 = 0.f, diff = 0.f;
    bool ofin = true, o01 = true;
    for (size_t i = 0; i < out.size(); ++i) {
        ofin = ofin && std::isfinite(out[i]);
        o01 = o01 && out[i] >= 0.f && out[i] <= 1.f;
        o2 += out[i] * out[i];
        float d = out[i] - rgb[i];
        diff += d * d;
    }
    CHECK(ofin && o01);
    CHECK(o2 > 0.f);
    CHECK(diff > 1e-6f);

    std::vector<float> zalt(z.size());
    for (size_t i = 0; i < z.size(); ++i)
        zalt[i] = z[i] + 0.4f;
    std::vector<float> out2(out.size());
    vae.decode(zalt.data(), g, out2.data());
    float d2 = 0.f;
    for (size_t i = 0; i < out.size(); ++i)
        d2 += (out[i] - out2[i]) * (out[i] - out2[i]);
    CHECK(d2 > 1e-6f);

    std::string dir = tmpdir();
    CHECK(h3_write_ppm(dir + "/f.ppm", out.data(), F, Ht, Wt, err) == Status::Ok);
    std::ifstream in(dir + "/f.ppm", std::ios::binary);
    std::string magic;
    in >> magic;
    CHECK(magic == "P6");

    CHECK(h3_vae_decoded_t(0, 5) == 3);
    CHECK(h3_vae_decoded_t(1, 5) == 4);
    CHECK(h3_vae_decoded_t(17, kH3VaeFirstChunkFrames) == 17 + kH3VaeFrameOffset + 3);
    CHECK(h3_vae_decoded_t(16, kH3VaeFirstChunkFrames) == 16 + kH3VaeFrameOffset);
    CHECK(h3_vae_tile_count(32) == 1);
    CHECK(h3_vae_tile_count(480) > 1);
    CHECK(h3_vae_tile_count(864) > 1);

    const int pad_t = 7, olh = 2, olw = 2;
    const int P = kH3VaeOutPatch;
    std::vector<float> rows(static_cast<size_t>(pad_t) * olh * olw * P, 0.f);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = static_cast<float>(i % 97) / 100.f;
    float im[3] = {0.f, 0.f, 0.f}, is[3] = {1.f, 1.f, 1.f};
    std::vector<float> u3(static_cast<size_t>(5) * 32 * 32 * 3, 0.f),
        u0(static_cast<size_t>(5) * 32 * 32 * 3, 0.f);
    h3_vae_unpack_3072(rows.data(), pad_t, olh, olw, 5, 32, 32, im, is, kH3VaeFrameOffset, u3.data());
    h3_vae_unpack_3072(rows.data(), pad_t, olh, olw, 5, 32, 32, im, is, 0, u0.data());
    float ud = 0.f;
    bool ufin = true;
    for (size_t i = 0; i < u3.size(); ++i) {
        ufin = ufin && std::isfinite(u3[i]) && std::isfinite(u0[i]) && u3[i] >= 0.f &&
               u3[i] <= 1.f;
        ud += (u3[i] - u0[i]) * (u3[i] - u0[i]);
    }
    CHECK(ufin);
    CHECK(ud > 1e-6f);

    std::string odir = tmpdir();
    const int hid = 64;
    auto f32 = [](const std::vector<float> &v) {
        return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(v.data()),
                                    reinterpret_cast<const uint8_t *>(v.data() + v.size()));
    };
    std::vector<float> emb(static_cast<size_t>(hid) * 24, 0.05f), ebb(hid, 0.01f);
    std::vector<float> reg(static_cast<size_t>(kH3VaeRegisters) * hid, 0.02f);
    std::vector<float> prj(static_cast<size_t>(kH3VaeOutPatch) * hid, 0.f), prb(kH3VaeOutPatch, 0.f);
    for (int o = 0; o < kH3VaeOutPatch; ++o)
        for (int i = 0; i < hid; ++i)
            prj[static_cast<size_t>(o) * hid + i] = ((o + i) % 11 - 5) * 0.01f;
    using OT = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
    std::vector<OT> ots;
    ots.push_back({"decoder.x_embedder.weight", "F32", {hid, 24}, f32(emb)});
    ots.push_back({"decoder.x_embedder.bias", "F32", {hid}, f32(ebb)});
    ots.push_back({"decoder.register_tokens", "F32", {1, kH3VaeRegisters, hid}, f32(reg)});
    ots.push_back({"decoder.proj_out.weight", "F32", {kH3VaeOutPatch, hid}, f32(prj)});
    ots.push_back({"decoder.proj_out.bias", "F32", {kH3VaeOutPatch}, f32(prb)});
    write_file(odir + "/config.json", R"({"model_type":"minimax_h3"})");
    write_safetensors_file(odir + "/model.safetensors", ots);
    H3Vae ovae;
    CHECK(ovae.load(odir, h3, err) == Status::Ok);
    CHECK(ovae.from_checkpoint);
    CHECK(ovae.official_decode);
    CHECK(!ovae.official_encode);
    std::vector<float> oout(rgb.size(), 0.f);
    ovae.decode(z.data(), g, oout.data());
    float oo2 = 0.f;
    bool oof = true, oo01 = true;
    for (float v : oout) {
        oof = oof && std::isfinite(v);
        oo01 = oo01 && v >= 0.f && v <= 1.f;
        oo2 += v * v;
    }
    CHECK(oof && oo01);
    CHECK(oo2 > 0.f);
    std::vector<float> oalt(oout.size());
    ovae.decode(zalt.data(), g, oalt.data());
    float od2 = 0.f;
    for (size_t i = 0; i < oout.size(); ++i)
        od2 += (oout[i] - oalt[i]) * (oout[i] - oalt[i]);
    CHECK(od2 > 1e-6f);

    // Tiny official 6-level Conv3d encoder (ch {8,8,8,16,16,24}, groups=8).
    std::string edir = tmpdir();
    const int ech[6] = {8, 8, 8, 16, 16, 24};
    const int espace[6] = {2, 2, 2, 2, 1, 1};
    const int etime[6] = {1, 2, 2, 1, 1, 1};
    auto efill = [](int n, int seed, float s) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] =
                static_cast<float>(((i + 1) * 17 + seed) % 13 - 6) * s;
        return v;
    };
    auto eones = [](int n) { return std::vector<float>(static_cast<size_t>(n), 1.f); };
    auto ezero = [](int n) { return std::vector<float>(static_cast<size_t>(n), 0.f); };
    std::vector<OT> ets;
    auto add5 = [&](const std::string &name, int o, int i, int k, int seed) {
        auto w = efill(o * i * k * k * k, seed, 0.02f);
        auto b = efill(o, seed + 3, 0.01f);
        ets.push_back({name + ".weight", "F32", {o, i, k, k, k}, f32(w)});
        ets.push_back({name + ".bias", "F32", {o}, f32(b)});
    };
    auto addn = [&](const std::string &name, int c) {
        ets.push_back({name + ".weight", "F32", {c}, f32(eones(c))});
        ets.push_back({name + ".bias", "F32", {c}, f32(ezero(c))});
    };
    add5("encoder.conv_in", 8, 3, 3, 1);
    int eprev = 8;
    for (int L = 0; L < 6; ++L) {
        for (int b = 0; b < 2; ++b) {
            const int in_ch = b ? ech[L] : eprev;
            const std::string bp =
                "encoder.down." + std::to_string(L) + ".block." + std::to_string(b);
            addn(bp + ".norm1", in_ch);
            add5(bp + ".conv1", ech[L], in_ch, 3, 10 + L * 4 + b);
            addn(bp + ".norm2", ech[L]);
            add5(bp + ".conv2", ech[L], ech[L], 3, 20 + L * 4 + b);
            if (in_ch != ech[L])
                add5(bp + ".nin_shortcut", ech[L], in_ch, 1, 30 + L);
        }
        if (espace[L] * etime[L] > 1)
            add5("encoder.down." + std::to_string(L) + ".downsample.conv", ech[L], ech[L], 3,
                 40 + L);
        eprev = ech[L];
    }
    addn("encoder.norm_out", 24);
    add5("encoder.conv_out", 48, 24, 3, 50);
    add5("quant_conv", 48, 48, 1, 60);
    write_file(edir + "/config.json", R"({"model_type":"minimax_h3"})");
    write_safetensors_file(edir + "/model.safetensors", ets);
    H3Vae evae;
    CHECK(evae.load(edir, h3, err) == Status::Ok);
    CHECK(evae.from_checkpoint);
    CHECK(evae.official_encode);
    CHECK(!evae.official_decode);
    CHECK(h3_encoder_latent_t(5) == 2);
    std::vector<float> ze(z.size(), 0.f);
    evae.encode(rgb.data(), g, ze.data());
    float ze2 = 0.f;
    bool zef = true;
    for (float v : ze) {
        zef = zef && std::isfinite(v);
        ze2 += v * v;
    }
    CHECK(zef);
    CHECK(ze2 > 0.f);
    float zemix = 0.f;
    for (size_t i = 0; i < z.size(); ++i)
        zemix += (ze[i] - z[i]) * (ze[i] - z[i]);
    CHECK(zemix > 1e-6f);
    std::vector<float> rgb_alt(rgb.size());
    for (size_t i = 0; i < rgb.size(); ++i)
        rgb_alt[i] = 1.f - rgb[i];
    std::vector<float> zealt(ze.size(), 0.f);
    evae.encode(rgb_alt.data(), g, zealt.data());
    float zed = 0.f;
    bool zeaf = true;
    for (size_t i = 0; i < ze.size(); ++i) {
        zeaf = zeaf && std::isfinite(zealt[i]);
        zed += (ze[i] - zealt[i]) * (ze[i] - zealt[i]);
    }
    CHECK(zeaf);
    CHECK(zed > 1e-6f);
    std::vector<float> edec(rgb.size(), 0.f);
    evae.decode(ze.data(), g, edec.data());
    float ediff = 0.f;
    bool edfin = true, ed01 = true;
    for (size_t i = 0; i < edec.size(); ++i) {
        edfin = edfin && std::isfinite(edec[i]);
        ed01 = ed01 && edec[i] >= 0.f && edec[i] <= 1.f;
        float d = edec[i] - rgb[i];
        ediff += d * d;
    }
    CHECK(edfin && ed01);
    CHECK(ediff > 1e-6f);
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(float));
    return static_cast<uint16_t>(u >> 16);
}

static void max_abs_check(const float *a, const float *b, int n, float eps) {
    for (int i = 0; i < n; ++i) {
        CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
        CHECK(std::fabs(a[i] - b[i]) < eps);
    }
}

static void test_gpu_backend() {
    using namespace mvllm;
    using namespace mvllm::quant;

    CHECK(std::strstr(gpu::compiled(), "cpu") != nullptr);
    CHECK(parse_device("metal") == Device::Metal);
    CHECK(parse_device("cuda") == Device::Cuda);
    CHECK(parse_device("cpu") == Device::Cpu);
    CHECK(std::strcmp(device_name(Device::Metal), "metal") == 0);

    gpu::select(Device::Cpu);
    CHECK(gpu::device() == Device::Cpu);
    CHECK(std::strcmp(gpu::name(), "cpu") == 0);
    CHECK(!gpu::gpu_ready());

    const int O = 4, I = 64, S = 2;
    std::vector<float> W(static_cast<size_t>(O) * I), X(static_cast<size_t>(S) * I), Yc(static_cast<size_t>(S) * O),
        Yg(static_cast<size_t>(S) * O);
    for (int i = 0; i < O * I; ++i)
        W[static_cast<size_t>(i)] = ((i * 17) % 11 - 5) * 0.1f;
    for (int i = 0; i < S * I; ++i)
        X[static_cast<size_t>(i)] = ((i * 3) % 7 - 3) * 0.2f;
    matmul_f32(Yc.data(), X.data(), W.data(), S, I, O);
    gpu::gemm_f32(Yg.data(), X.data(), W.data(), S, I, O);
    max_abs_check(Yc.data(), Yg.data(), S * O, 1e-5f);

    std::vector<uint8_t> p4(static_cast<size_t>(O) * (I / 2));
    std::vector<float> s4(static_cast<size_t>(O) * (I / 64));
    quantize_int4_g64(W.data(), O, I, p4.data(), s4.data());
    matmul_int4_g64(Yc.data(), X.data(), p4.data(), s4.data(), S, I, O);
    gpu::gemm_int4_g64(Yg.data(), X.data(), p4.data(), s4.data(), S, I, O);
    max_abs_check(Yc.data(), Yg.data(), S * O, 1e-5f);

    std::vector<uint8_t> pm(static_cast<size_t>(O) * (I / 2)), sm(static_cast<size_t>(O) * (I / 32));
    pack_mxfp4(W.data(), O, I, pm.data(), sm.data());
    matmul_mxfp4(Yc.data(), X.data(), pm.data(), sm.data(), S, I, O);
    gpu::gemm_mxfp4(Yg.data(), X.data(), pm.data(), sm.data(), S, I, O, false);
    max_abs_check(Yc.data(), Yg.data(), S * O, 1e-5f);

    const int kI = 32, kO = 32;
    std::vector<float> W1(static_cast<size_t>(kO) * kI), W2(static_cast<size_t>(kI) * kO),
        W3(static_cast<size_t>(kO) * kI), Z(kI);
    for (int i = 0; i < kO * kI; ++i) {
        W1[static_cast<size_t>(i)] = ((i * 13) % 11 - 5) * 0.08f;
        W3[static_cast<size_t>(i)] = ((i * 19) % 9 - 4) * 0.07f;
    }
    for (int i = 0; i < kI * kO; ++i)
        W2[static_cast<size_t>(i)] = ((i * 11) % 7 - 3) * 0.06f;
    for (int i = 0; i < kI; ++i)
        Z[static_cast<size_t>(i)] = ((i * 5) % 9 - 4) * 0.2f;
    const int64_t w1p = static_cast<int64_t>(kO) * (kI / 2);
    const int64_t w1s = static_cast<int64_t>(kO) * (kI / 32);
    const int64_t w2p = static_cast<int64_t>(kI) * (kO / 2);
    const int64_t w2s = static_cast<int64_t>(kI) * (kO / 32);
    std::vector<uint8_t> kblob(static_cast<size_t>(2 * (w1p + w1s) + w2p + w2s));
    pack_mxfp4(W1.data(), kO, kI, kblob.data(), kblob.data() + w1p);
    pack_mxfp4(W2.data(), kI, kO, kblob.data() + w1p + w1s, kblob.data() + w1p + w1s + w2p);
    pack_mxfp4(W3.data(), kO, kI, kblob.data() + w1p + w1s + w2p + w2s,
               kblob.data() + w1p + w1s + w2p + w2s + w1p);
    std::vector<float> yk(kI);
    gpu::k3_expert(yk.data(), Z.data(), 1, kblob.data(), kI, kO, 4.f, 25.f, false);
    for (float v : yk)
        CHECK(std::isfinite(v));

    const int gH = 64, gO = 64;
    std::vector<float> G(static_cast<size_t>(gO) * gH), U(static_cast<size_t>(gO) * gH),
        D(static_cast<size_t>(gH) * gO), Xg(gH);
    for (int i = 0; i < gO * gH; ++i) {
        G[static_cast<size_t>(i)] = ((i * 7) % 11 - 5) * 0.05f;
        U[static_cast<size_t>(i)] = ((i * 5) % 9 - 4) * 0.05f;
    }
    for (int i = 0; i < gH * gO; ++i)
        D[static_cast<size_t>(i)] = ((i * 3) % 7 - 3) * 0.04f;
    for (int i = 0; i < gH; ++i)
        Xg[static_cast<size_t>(i)] = ((i * 2) % 5 - 2) * 0.1f;
    const int64_t pack = static_cast<int64_t>(gO) * gH / 2;
    const int64_t sc = static_cast<int64_t>(gO) * gH / 64 * 4;
    std::vector<uint8_t> gblob(static_cast<size_t>(3 * (pack + sc)));
    quantize_int4_g64(G.data(), gO, gH, gblob.data(), reinterpret_cast<float *>(gblob.data() + pack));
    quantize_int4_g64(U.data(), gO, gH, gblob.data() + pack + sc,
                      reinterpret_cast<float *>(gblob.data() + pack + sc + pack));
    quantize_int4_g64(D.data(), gH, gO, gblob.data() + 2 * (pack + sc),
                      reinterpret_cast<float *>(gblob.data() + 2 * (pack + sc) + pack));
    std::vector<float> yg(gH);
    gpu::glm_expert(yg.data(), Xg.data(), 1, gblob.data(), gH, gO, 7.f);
    for (float v : yg)
        CHECK(std::isfinite(v));

    const int H = 8, Inn = 4, Ffn = 8, T = 3, hd = 2;
    const int64_t qkv_n = static_cast<int64_t>(3) * Inn * H;
    const int64_t out_n = static_cast<int64_t>(H) * Inn;
    const int64_t fc1_n = static_cast<int64_t>(2) * Ffn * H;
    const int64_t fc2_n = static_cast<int64_t>(H) * Ffn;
    std::vector<uint8_t> dblob(static_cast<size_t>(2 * (qkv_n + out_n + fc1_n + fc2_n)));
    uint16_t *bf = reinterpret_cast<uint16_t *>(dblob.data());
    for (int64_t i = 0; i < qkv_n + out_n + fc1_n + fc2_n; ++i)
        bf[i] = f32_to_bf16(((i * 17) % 11 - 5) * 0.05f);
    std::vector<float> xc(static_cast<size_t>(T) * H), xg(static_cast<size_t>(T) * H);
    for (int i = 0; i < T * H; ++i)
        xc[static_cast<size_t>(i)] = xg[static_cast<size_t>(i)] = ((i % 7) - 3) * 0.1f;
    h3_dit_block_cpu(dblob.data(), qkv_n * 2, out_n * 2, fc1_n * 2, fc2_n * 2, H, Inn, Ffn, hd,
                     xc.data(), T, 1e-6f);
    gpu::dit_block(dblob.data(), qkv_n * 2, out_n * 2, fc1_n * 2, fc2_n * 2, H, Inn, Ffn, hd,
                   xg.data(), T, 1e-6f);
    max_abs_check(xc.data(), xg.data(), T * H, 1e-5f);

    gpu::select(Device::Metal);
#if defined(MVLLM_WITH_METAL)
    CHECK(std::strstr(gpu::compiled(), "metal") != nullptr);
#endif
    if (gpu::gpu_ready()) {
        std::vector<float> Ym(static_cast<size_t>(S) * O);
        matmul_f32(Yc.data(), X.data(), W.data(), S, I, O);
        gpu::gemm_f32(Ym.data(), X.data(), W.data(), S, I, O);
        max_abs_check(Yc.data(), Ym.data(), S * O, 2e-4f);

        gpu::gemm_int4_g64(Ym.data(), X.data(), p4.data(), s4.data(), S, I, O);
        matmul_int4_g64(Yc.data(), X.data(), p4.data(), s4.data(), S, I, O);
        max_abs_check(Yc.data(), Ym.data(), S * O, 2e-4f);

        gpu::gemm_mxfp4(Ym.data(), X.data(), pm.data(), sm.data(), S, I, O, false);
        matmul_mxfp4(Yc.data(), X.data(), pm.data(), sm.data(), S, I, O);
        max_abs_check(Yc.data(), Ym.data(), S * O, 2e-4f);

        std::vector<float> yk_g(kI);
        gpu::select(Device::Cpu);
        gpu::k3_expert(yk.data(), Z.data(), 1, kblob.data(), kI, kO, 4.f, 25.f, false);
        gpu::select(Device::Metal);
        gpu::k3_expert(yk_g.data(), Z.data(), 1, kblob.data(), kI, kO, 4.f, 25.f, false);
        max_abs_check(yk.data(), yk_g.data(), kI, 2e-3f);

        std::vector<float> yg_g(gH);
        gpu::select(Device::Cpu);
        gpu::glm_expert(yg.data(), Xg.data(), 1, gblob.data(), gH, gO, 7.f);
        gpu::select(Device::Metal);
        gpu::glm_expert(yg_g.data(), Xg.data(), 1, gblob.data(), gH, gO, 7.f);
        max_abs_check(yg.data(), yg_g.data(), gH, 2e-3f);

        std::vector<float> xd(static_cast<size_t>(T) * H);
        for (int i = 0; i < T * H; ++i)
            xd[static_cast<size_t>(i)] = ((i % 7) - 3) * 0.1f;
        gpu::dit_block(dblob.data(), qkv_n * 2, out_n * 2, fc1_n * 2, fc2_n * 2, H, Inn, Ffn, hd,
                       xd.data(), T, 1e-6f);
        max_abs_check(xc.data(), xd.data(), T * H, 2e-3f);

        std::string dir = tmpdir();
        write_file(dir + "/config.json",
                   R"({"model_type":"minimax_h3","architectures":["MiniMaxH3"]})");
        using Tup = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
        std::vector<Tup> ts;
        auto bf16z = [&](size_t n) { return std::vector<uint8_t>(n * 2, 0); };
        for (int b = 0; b < 2; ++b) {
            std::string p = "blocks." + std::to_string(b) + ".";
            ts.push_back({p + "attn.qkv_proj.weight", "BF16", {Inn * 3, H},
                          bf16z(static_cast<size_t>(Inn * 3 * H))});
            ts.push_back({p + "attn.out_proj.weight", "BF16", {H, Inn},
                          bf16z(static_cast<size_t>(H * Inn))});
            ts.push_back({p + "mlp.fc1.weight", "BF16", {Ffn * 2, H},
                          bf16z(static_cast<size_t>(Ffn * 2 * H))});
            ts.push_back({p + "mlp.fc2.weight", "BF16", {H, Ffn},
                          bf16z(static_cast<size_t>(H * Ffn))});
        }
        write_safetensors_file(dir + "/model.safetensors", ts);
        Engine eh;
        RuntimeConfig rt;
        rt.device = Device::Metal;
        std::string err;
        CHECK(eh.load(dir, rt, err) == Status::Ok);
        CHECK(eh.info().find("gpu=metal") != std::string::npos);
        H3GenParams hp;
        hp.steps = 1;
        hp.dit_layers = 2;
        hp.width = 32;
        hp.height = 32;
        hp.frames = 5;
        hp.output_path = dir + "/out.txt";
        H3GenResult hr;
        CHECK(eh.generate_video(hp, hr, err) == Status::Ok);
        CHECK(hr.note.find("Metal DiT") != std::string::npos);
    } else {
        gpu::select(Device::Metal);
        CHECK(gpu::device() == Device::Cpu);
    }

    gpu::select(Device::Cpu);
    CHECK(gpu::device() == Device::Cpu);
}

static void test_sample_and_rope() {
    using namespace mvllm;
    float logits[4] = {0.f, 3.f, 1.f, 2.f};
    uint64_t rng = 1;
    CHECK(sample_token(logits, 4, 0.f, 1.f, &rng) == 1);
    int t = sample_token(logits, 4, 0.8f, 0.9f, &rng);
    CHECK(t >= 0 && t < 4);
    float x[4] = {1.f, 0.f, 0.f, 1.f};
    apply_rope(x, 4, 0, 10000.f);
    CHECK_NEAR(x[0], 1.f, 1e-5);
    CHECK_NEAR(x[1], 0.f, 1e-5);
    float y[4] = {1.f, 0.f, 0.f, 1.f};
    apply_rope(y, 4, 3, 10000.f);
    CHECK(std::fabs(y[0] - 1.f) > 1e-4f);

    KdaConfig kda;
    kda.heads = 1;
    kda.head_dim = 4;
    kda.full_rank_gate = false;
    std::vector<float> ga_w(4 * 8, 0.1f), gb_w(4 * 4, 0.2f), go_w(8 * 4, 0.05f);
    std::vector<float> qw(4 * 8, 0.15f), kw(4 * 8, 0.12f), vw(4 * 8, 0.18f);
    quant::QuantMat ga, gb, go, wq, wk, wv;
    ga.from_f32(ga_w.data(), 4, 8, 32);
    gb.from_f32(gb_w.data(), 4, 4, 32);
    go.from_f32(go_w.data(), 8, 4, 32);
    wq.from_f32(qw.data(), 4, 8, 32);
    wk.from_f32(kw.data(), 4, 8, 32);
    wv.from_f32(vw.data(), 4, 8, 32);
    std::vector<float> xin(8, 0.3f), S(16, 0.f), y0(8, 0.f), y1(8, 0.f), on(4, 1.f);
    kda_step(xin.data(), 8, kda, &wq, &wk, &wv, nullptr, nullptr, nullptr, nullptr, 0, nullptr, &ga,
             nullptr, &go, on.data(), S.data(), y0.data(), 1e-6f);
    std::fill(S.begin(), S.end(), 0.f);
    kda_step(xin.data(), 8, kda, &wq, &wk, &wv, nullptr, nullptr, nullptr, nullptr, 0, nullptr, &ga,
             &gb, &go, on.data(), S.data(), y1.data(), 1e-6f);
    float gd = 0.f;
    for (int i = 0; i < 8; ++i) {
        CHECK(std::isfinite(y0[i]) && std::isfinite(y1[i]));
        gd += (y0[i] - y1[i]) * (y0[i] - y1[i]);
    }
    CHECK(gd > 1e-10f);

    VisionConfig v;
    v.patch = 4;
    v.image_size = 8;
    v.merge = 1;
    v.hidden = 8;
    v.out_hidden = 8;
    std::vector<float> rgb(static_cast<size_t>(8 * 8 * 3), 0.5f);
    std::vector<float> pw(static_cast<size_t>(8 * 3 * 4 * 4), 0.02f), pr(static_cast<size_t>(8 * 8), 0.1f);
    for (size_t i = 0; i < pw.size(); ++i)
        pw[i] = ((static_cast<int>(i) % 5) - 2) * 0.03f;
    quant::QuantMat patch, proj;
    patch.from_f32(pw.data(), 8, 3 * 4 * 4, 32);
    proj.from_f32(pr.data(), 8, 8, 32);
    std::vector<float> vout(static_cast<size_t>(4 * 8), 0.f);
    int ntok = glm_vit_embed(rgb.data(), 8, 8, v, &patch, &proj, vout.data(), 4);
    CHECK(ntok == 4);
    float vn = 0.f;
    for (float a : vout) {
        CHECK(std::isfinite(a));
        vn += a * a;
    }
    CHECK(vn > 0.f);
    CHECK(glm_vit_embed(nullptr, 8, 8, v, &patch, &proj, vout.data(), 4) == 0);

    GlmVitTower tw;
    tw.cfg = v;
    tw.cfg.heads = 2;
    tw.cfg.hidden = 8;
    tw.cfg.out_hidden = 8;
    tw.cfg.merge = 2;
    tw.cfg.temporal = 1;
    tw.cfg.in_channels = 3;
    tw.cfg.intermediate = 8;
    tw.cfg.proj_intermediate = 8;
    tw.cfg.patch = 4;
    const int pin = 3 * 1 * 4 * 4;
    tw.patch_w.assign(static_cast<size_t>(8 * pin), 0.05f);
    GlmVitBlock blk;
    blk.norm1.assign(8, 1.f);
    blk.norm2.assign(8, 1.f);
    blk.qkv_w.assign(static_cast<size_t>(24 * 8), 0.02f);
    blk.q_norm.assign(4, 1.f);
    blk.k_norm.assign(4, 1.f);
    blk.proj_w.assign(static_cast<size_t>(8 * 8), 0.03f);
    blk.gate_w.assign(static_cast<size_t>(8 * 8), 0.02f);
    blk.up_w.assign(static_cast<size_t>(8 * 8), 0.02f);
    blk.down_w.assign(static_cast<size_t>(8 * 8), 0.02f);
    tw.blocks.push_back(blk);
    tw.post_norm.assign(8, 1.f);
    std::vector<float> pix(static_cast<size_t>(4 * pin), 0.1f);
    std::vector<float> tout(8, 0.f);
    CHECK(glm_vit_forward(tw, pix.data(), 2, 2, tout.data()) == 1);
    float tn = 0.f;
    for (float a : tout) {
        CHECK(std::isfinite(a));
        tn += a * a;
    }
    CHECK(tn > 0.f);
    std::vector<float> rgb2(static_cast<size_t>(8 * 8 * 3), 0.4f);
    std::vector<float> e2(8, 0.f);
    CHECK(glm_vit_embed(rgb2.data(), 8, 8, tw.cfg, nullptr, nullptr, e2.data(), 1, &tw) == 1);

    const int ch = 2, tt = 2, hh = 2, ww = 2;
    std::vector<float> zz(static_cast<size_t>(ch * tt * hh * ww));
    for (size_t i = 0; i < zz.size(); ++i)
        zz[i] = static_cast<float>(i);
    std::vector<float> prow(zz.size()), back(zz.size());
    CHECK(h3_dit_patchify(zz.data(), ch, tt, hh, ww, prow.data()) == static_cast<int>(zz.size()));
    CHECK(h3_dit_unpatchify(prow.data(), ch, tt, hh, ww, back.data()) == static_cast<int>(zz.size()));
    for (size_t i = 0; i < zz.size(); ++i)
        CHECK_NEAR(back[i], zz[i], 1e-6);
    float sig[5];
    h3_sigma_video(4, sig, 12.f);
    CHECK(sig[0] > sig[1] && sig[3] > sig[4] && sig[4] == 0.f);
    float samp[2] = {1.f, 2.f}, vel[2] = {1.f, -1.f};
    CHECK(h3_euler_step(samp, vel, 2, 1.f, 0.5f) == 1);
    CHECK_NEAR(samp[0], 1.5f, 1e-5);
    CHECK_NEAR(samp[1], 1.5f, 1e-5);
    float tf[8];
    h3_time_features(0.3f, tf, 8);
    CHECK(std::isfinite(tf[0]) && std::isfinite(tf[4]));
}

int main() {
    test_quant();
    test_quant_mat();
    test_idot();
    test_expert_store();
    test_block_store();
    test_safetensors();
    test_http_helpers();
    test_tokenizer();
    test_k3_xtml();
    test_tiktoken_model();
    test_glm_eos_ids();
    test_config_and_families();
    test_dsv4_tiny();
    test_dsv4_prefix_ckpt();
    test_dsv4_cuda_tier();
    test_coli_cuda_tier();
    test_vk_ops_tier();
    test_metal_ops_tier();
    test_metal_h3_tier();
    test_llama_dims_helpers();
    test_offload_generate();
    test_glm53_container();
    test_k3_mxfp4_container();
    test_h3_checkpoint();
    test_kda_short_conv();
    test_dsa();
    test_shard_probe();
    test_live_dump_env();
    test_moe_union();
    test_dense_bits_generate();
    test_mla_absorb();
    test_mla_generate();
    test_h3_vae();
    {
        using namespace mvllm;
        CHECK(h3_audio_t(5) == 8);
        CHECK(h3_audio_t(56) == 93);
        CHECK(h3_audio_pad_samples(1) == 800);
        CHECK(h3_audio_pad_samples(800) == 800);
        CHECK(h3_audio_samples(2) == 1600);
        enum { AC = 4, AT = 3, AUDIO = AC * 2 * AT };
        float audio[AUDIO], packed[AUDIO], unpacked[AUDIO];
        for (int i = 0; i < AUDIO; ++i)
            audio[i] = static_cast<float>(i);
        CHECK(h3_dit_pack_audio(audio, AC, AT, packed) == AUDIO);
        const float expected[] = {0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20,
                                  3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23};
        for (int i = 0; i < AUDIO; ++i)
            CHECK_NEAR(packed[i], expected[i], 1e-6);
        CHECK(h3_dit_unpack_audio(packed, AC, AT, unpacked) == AUDIO);
        for (int i = 0; i < AUDIO; ++i)
            CHECK_NEAR(unpacked[i], audio[i], 1e-6);

        H3AudioVae av;
        std::string aerr;
        CHECK(av.load("/tmp/does-not-exist-mvllm-avae", aerr) == Status::Ok);
        CHECK(!av.from_checkpoint);
        const int samples = 1600;
        std::vector<float> pcm(static_cast<size_t>(2) * samples);
        for (int t = 0; t < samples; ++t) {
            pcm[static_cast<size_t>(t)] = 0.2f * std::sin(0.04f * static_cast<float>(t));
            pcm[static_cast<size_t>(samples + t)] = 0.15f * std::cos(0.03f * static_cast<float>(t));
        }
        std::vector<float> z;
        int at = 0;
        av.encode(pcm.data(), samples, z, at);
        CHECK(at == 2);
        CHECK(static_cast<int>(z.size()) == kH3AudioChannels * 2 * at);
        float z2 = 0.f;
        bool zfin = true;
        for (float v : z) {
            zfin = zfin && std::isfinite(v);
            z2 += v * v;
        }
        CHECK(zfin);
        CHECK(z2 > 0.f);
        std::vector<float> outp;
        av.decode(z.data(), at, outp);
        CHECK(static_cast<int>(outp.size()) == 2 * at * kH3AudioHop);
        float o2 = 0.f, diff = 0.f;
        bool ofin = true, oclip = true;
        for (size_t i = 0; i < outp.size(); ++i) {
            ofin = ofin && std::isfinite(outp[i]);
            oclip = oclip && outp[i] >= -1.f && outp[i] <= 1.f;
            o2 += outp[i] * outp[i];
            if (i < pcm.size()) {
                float d = outp[i] - pcm[i];
                diff += d * d;
            }
        }
        CHECK(ofin && oclip);
        CHECK(o2 > 0.f);
        CHECK(diff > 1e-6f);
        std::vector<float> zalt = z;
        for (float &v : zalt)
            v += 0.5f;
        std::vector<float> out2;
        av.decode(zalt.data(), at, out2);
        float d2 = 0.f;
        for (size_t i = 0; i < outp.size() && i < out2.size(); ++i)
            d2 += (outp[i] - out2[i]) * (outp[i] - out2[i]);
        CHECK(d2 > 1e-6f);

        std::string dir = tmpdir();
        CHECK(h3_write_wav(dir + "/a.wav", pcm.data(), 2, samples, kH3AudioRate, aerr) == Status::Ok);
        std::vector<float> back;
        int ch = 0, sn = 0, rt = 0;
        CHECK(h3_read_wav(dir + "/a.wav", back, ch, sn, rt, aerr) == Status::Ok);
        CHECK(ch == 2 && sn == samples && rt == kH3AudioRate);
        CHECK(std::fabs(back[10] - pcm[10]) < 2e-4f);
    }
    {
        using namespace mvllm;
        H3Layout lay;
        CHECK(h3_layout_build(12, 2, 2, 2, 8, 5, lay));
        CHECK(lay.text_rows == 12);
        CHECK(lay.audio_rows == 16);
        CHECK(lay.video_rows == 2);
        CHECK(lay.seq_len == 30);
        CHECK(lay.segments.size() == 3);
        CHECK(lay.segments[0].kind == H3SegKind::Text);
        CHECK(lay.segments[1].kind == H3SegKind::Audio);
        CHECK(lay.segments[2].kind == H3SegKind::Video);
        CHECK_NEAR(lay.positions[0].t, 0.f, 1e-5);
        CHECK_NEAR(lay.positions[1].t, 1.f, 1e-5);
        float inv[kH3RopeFreqs];
        h3_dit_default_inv_freq(inv);
        std::vector<float> cs, sn;
        h3_dit_rope_tables(lay, inv, 1.f, cs, sn);
        CHECK(static_cast<int>(cs.size()) == 30 * kH3RopeHalf);
        CHECK_NEAR(cs[0], 1.f, 1e-5);
        CHECK_NEAR(sn[0], 0.f, 1e-5);
        // row 1 is text t=1 → axis 0, freq 0: angle = 1 * inv[0]
        CHECK(std::fabs(cs[kH3RopeHalf] - std::cos(inv[0])) < 1e-5f);
        float q[256], k[256];
        for (int i = 0; i < 256; ++i) {
            q[i] = 0.01f * static_cast<float>(i);
            k[i] = 0.02f * static_cast<float>(i);
        }
        float q0 = q[0], q48 = q[48];
        h3_dit_apply_rope(q, k, cs.data(), sn.data(), 1, 1, 128);
        CHECK(std::fabs(q[0] - (q0 * cs[0] - q48 * sn[0])) < 1e-5f);
    }
    {
        using namespace mvllm;
        if (h3_ffmpeg_available()) {
            std::vector<float> rgb(static_cast<size_t>(5) * 32 * 32 * 3, 0.2f);
            std::vector<float> pcm(static_cast<size_t>(2) * 1600, 0.1f);
            std::string dir = tmpdir();
            std::string err;
            CHECK(h3_write_mp4(dir + "/clip.mp4", rgb.data(), 5, 32, 32, 24, pcm.data(), 2, 1600,
                               32000, err) == Status::Ok);
            std::ifstream in(dir + "/clip.mp4", std::ios::binary);
            char mag[8] = {};
            in.read(mag, 8);
            CHECK(in.gcount() >= 8);
            CHECK(std::memcmp(mag + 4, "ftyp", 4) == 0);
        } else {
            std::string err;
            std::vector<float> rgb(12, 0.f);
            CHECK(h3_write_mp4("/tmp/no.mp4", rgb.data(), 1, 2, 2, 24, nullptr, 0, 0, 0, err) ==
                  Status::Unsupported);
        }
    }
    test_gpu_backend();
    test_sample_and_rope();
    {
        using namespace mvllm;
        H3TextEncoder enc;
        std::string e;
        CHECK(enc.load("/tmp/does-not-exist-mvllm-text", e) == Status::Ok);
        CHECK(enc.ready());
        CHECK(!enc.from_checkpoint());
        std::vector<int> ids;
        h3_text_ids_from_prompt("a red fox", enc.config().vocab, ids);
        CHECK(!ids.empty());
        std::vector<float> hid;
        enc.encode(ids, hid);
        CHECK(static_cast<int>(hid.size()) == static_cast<int>(ids.size()) * enc.config().hidden);
        for (float v : hid)
            CHECK(std::isfinite(v));
    }
    {
        using namespace mvllm;
        H3Layout lay;
        CHECK(h3_layout_build(12, 2, 2, 2, 8, 5, lay));
        CHECK(lay.seq_len == 30);
        CHECK(lay.img_cond_rows == 0);
        int kf0[] = {0};
        H3Layout klay;
        CHECK(h3_layout_build(12, 2, 2, 2, 8, 5, klay, kf0, 1, nullptr, 0));
        CHECK(klay.img_cond_rows == 1);
        CHECK(klay.seq_len == 31);
        CHECK(klay.segments.size() == 4);
        CHECK(klay.segments[1].kind == H3SegKind::Cond);
        H3LayoutRef ref;
        ref.kind = H3SegKind::RefImage;
        ref.latent_h = 2;
        ref.latent_w = 2;
        H3Layout rlay;
        CHECK(h3_layout_build(12, 2, 2, 2, 8, 5, rlay, nullptr, 0, &ref, 1));
        CHECK(rlay.img_cond_rows == 1);
        CHECK(rlay.segments[1].kind == H3SegKind::RefImage);
        CHECK(rlay.seq_len == 31);
        H3Layout bad;
        CHECK(!h3_layout_build(12, 2, 2, 2, 8, 5, bad, kf0, 1, &ref, 1));
        int kf_mid[] = {2};
        CHECK(!h3_layout_build(12, 2, 2, 2, 8, 5, bad, kf_mid, 1, nullptr, 0));

        H3VisionOut vo;
        vo.grid_h = 2;
        vo.grid_w = 2;
        vo.tokens = 1;
        vo.out_width = 32;
        vo.merged.assign(32, 0.3f);
        vo.deepstack[0].assign(32, 0.01f);
        H3MmSeq seq;
        CHECK(h3_mm_build_fl2va("a red fox", &vo, 1, nullptr, 256, seq));
        CHECK(!seq.ids.empty());
        CHECK(seq.spans.size() == 1);
        CHECK(seq.spans[0].tokens == 1);
        bool saw_start = false, saw_pad = false, saw_end = false;
        for (int id : seq.ids) {
            saw_start = saw_start || id == static_cast<int>(kH3VisionStart);
            saw_pad = saw_pad || id == static_cast<int>(kH3ImagePad);
            saw_end = saw_end || id == static_cast<int>(kH3VisionEnd);
        }
        CHECK(saw_start && saw_pad && saw_end);
        CHECK(seq.tags[static_cast<size_t>(seq.spans[0].start)] == 0);
        CHECK(static_cast<int>(seq.positions.size()) == 3 * static_cast<int>(seq.ids.size()));
        H3RefPres pres;
        pres.kind = H3PresKind::Image;
        pres.vision = &vo;
        pres.vision_count = 1;
        H3MmSeq rseq;
        CHECK(h3_mm_build_ref2va("a red fox", &pres, 1, nullptr, 256, rseq));
        CHECK(rseq.spans.size() == 1);
        H3RefPres audio_only;
        audio_only.kind = H3PresKind::Audio;
        audio_only.has_audio = true;
        H3MmSeq aseq;
        CHECK(!h3_mm_build_ref2va("a red fox", &audio_only, 1, nullptr, 256, aseq));

        H3TextEncoder enc2;
        std::string e2;
        CHECK(enc2.load("/tmp/does-not-exist-mvllm-text", e2) == Status::Ok);
        std::vector<float> mmhid, plain;
        enc2.encode(seq.ids, plain);
        enc2.encode_mm(seq.ids, seq.spans.data(), static_cast<int>(seq.spans.size()),
                       seq.positions.data(), seq.tags.data(), mmhid);
        CHECK(mmhid.size() == plain.size());
        float md = 0.f;
        bool mfin = true;
        for (size_t i = 0; i < mmhid.size(); ++i) {
            mfin = mfin && std::isfinite(mmhid[i]);
            md += (mmhid[i] - plain[i]) * (mmhid[i] - plain[i]);
        }
        CHECK(mfin);
        CHECK(md > 1e-8f);
        std::vector<float> oneL, allL;
        enc2.encode_mm(seq.ids, seq.spans.data(), static_cast<int>(seq.spans.size()),
                       seq.positions.data(), seq.tags.data(), oneL, 1);
        enc2.encode_mm(seq.ids, seq.spans.data(), static_cast<int>(seq.spans.size()),
                       seq.positions.data(), seq.tags.data(), allL, 0);
        CHECK(oneL.size() == allL.size());
        float ld = 0.f;
        for (size_t i = 0; i < oneL.size(); ++i)
            ld += (oneL[i] - allL[i]) * (oneL[i] - allL[i]);
        if (enc2.config().layers > 1)
            CHECK(ld > 1e-10f);

        std::string vdir = tmpdir();
        const int vh = 32, vI = 64, vL = 2, vout = 32;
        const int pdim = 3 * 2 * 16 * 16;
        auto vf = [](int n, int seed, float s) {
            std::vector<float> v(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
                v[static_cast<size_t>(i)] =
                    static_cast<float>(((i + 1) * 13 + seed) % 11 - 5) * s;
            return v;
        };
        auto vb = [](const std::vector<float> &v) {
            return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(v.data()),
                                        reinterpret_cast<const uint8_t *>(v.data() + v.size()));
        };
        using VT = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
        std::vector<VT> vts;
        auto add2 = [&](const std::string &n, int o, int i, int seed) {
            auto w = vf(o * i, seed, 0.03f);
            auto b = vf(o, seed + 1, 0.01f);
            vts.push_back({n + ".weight", "F32", {o, i}, vb(w)});
            vts.push_back({n + ".bias", "F32", {o}, vb(b)});
        };
        auto add1 = [&](const std::string &n, int o) {
            vts.push_back({n + ".weight", "F32", {o}, vb(std::vector<float>(static_cast<size_t>(o), 1.f))});
            vts.push_back({n + ".bias", "F32", {o}, vb(std::vector<float>(static_cast<size_t>(o), 0.f))});
        };
        {
            auto w = vf(vh * pdim, 1, 0.02f);
            auto b = vf(vh, 2, 0.01f);
            vts.push_back({"model.visual.patch_embed.proj.weight", "F32", {vh, 3, 2, 16, 16}, vb(w)});
            vts.push_back({"model.visual.patch_embed.proj.bias", "F32", {vh}, vb(b)});
        }
        for (int l = 0; l < vL; ++l) {
            const std::string B = "model.visual.blocks." + std::to_string(l) + ".";
            add1(B + "norm1", vh);
            add2(B + "attn.qkv", vh * 3, vh, 10 + l);
            add2(B + "attn.proj", vh, vh, 20 + l);
            add1(B + "norm2", vh);
            add2(B + "mlp.linear_fc1", vI, vh, 30 + l);
            add2(B + "mlp.linear_fc2", vh, vI, 40 + l);
        }
        add1("model.visual.merger.norm", vh);
        add2("model.visual.merger.linear_fc1", vh * 4, vh * 4, 50);
        add2("model.visual.merger.linear_fc2", vout, vh * 4, 60);
        write_file(vdir + "/config.json", R"({"model_type":"minimax_h3"})");
        write_safetensors_file(vdir + "/model.safetensors", vts);
        H3VisionEncoder venc;
        std::string verr;
        CHECK(venc.load(vdir, verr) == Status::Ok);
        CHECK(venc.from_checkpoint());
        CHECK(venc.ready());
        CHECK(venc.config().hidden == vh);
        CHECK(venc.config().layers == vL);
        std::vector<float> rgb(static_cast<size_t>(32) * 32 * 3);
        for (size_t i = 0; i < rgb.size(); ++i)
            rgb[i] = (i % 13) / 12.f;
        H3VisionOut e1, ealt;
        venc.encode(rgb.data(), 1, 32, 32, e1);
        CHECK(e1.tokens == 1);
        CHECK(e1.grid_h == 2 && e1.grid_w == 2);
        CHECK(static_cast<int>(e1.merged.size()) == e1.tokens * e1.out_width);
        float e12 = 0.f;
        bool efin = true;
        for (float v : e1.merged) {
            efin = efin && std::isfinite(v);
            e12 += v * v;
        }
        CHECK(efin);
        CHECK(e12 > 0.f);
        std::vector<float> rgb2 = rgb;
        for (float &v : rgb2)
            v = 1.f - v;
        venc.encode(rgb2.data(), 1, 32, 32, ealt);
        float ed = 0.f;
        for (size_t i = 0; i < e1.merged.size() && i < ealt.merged.size(); ++i)
            ed += (e1.merged[i] - ealt.merged[i]) * (e1.merged[i] - ealt.merged[i]);
        CHECK(ed > 1e-6f);
        H3VisionOut badg;
        venc.encode(rgb.data(), 1, 31, 32, badg);
        CHECK(badg.tokens == 0);

        auto h3e = make_engine(Family::H3);
        RuntimeConfig rt;
        std::string herr;
        std::string hdir = tmpdir();
        write_file(hdir + "/config.json", R"({"model_type":"minimax_h3"})");
        CHECK(h3e->load(hdir, rt, herr) == Status::Ok);
        H3GenParams hp;
        hp.prompt = "a red fox";
        hp.width = 32;
        hp.height = 32;
        hp.frames = 5;
        hp.steps = 1;
        hp.dit_layers = 1;
        hp.ref_images.push_back("x.png");
        hp.first_frame = "y.png";
        hp.output_path = hdir + "/out.txt";
        H3GenResult hout;
        CHECK(h3e->generate_video(hp, hout, herr) != Status::Ok);
        hp.first_frame.clear();
        hp.ref_images.clear();
        hp.ref_rgb = rgb.data();
        hp.ref_w = 32;
        hp.ref_h = 32;
        herr.clear();
        CHECK(h3e->generate_video(hp, hout, herr) == Status::Ok);
        CHECK(hout.note.find("ref2va") != std::string::npos ||
              hout.note.find("picture=") != std::string::npos);
    }
    {
        using namespace mvllm;
        Gbnf g;
        std::string e;
        CHECK(g.compile("root ::= \"yes\"", e) == Status::Ok);
        CHECK(g.ready());
        auto dec = [](int id) -> std::string {
            if (id == 1)
                return "y";
            if (id == 2)
                return "e";
            if (id == 3)
                return "s";
            if (id == 4)
                return "n";
            return "";
        };
        std::vector<uint8_t> ok(5, 0);
        g.allow_mask(dec, ok.data(), 5);
        CHECK(ok[1] == 1);
        CHECK(ok[4] == 0);
        CHECK(g.accept_bytes("ye"));
        g.allow_mask(dec, ok.data(), 5);
        CHECK(ok[3] == 1);
        CHECK(ok[1] == 0);
        Gbnf g2;
        CHECK(g2.compile("root ::= [0-9]+", e) == Status::Ok);
        auto decd = [](int id) -> std::string {
            if (id == 1)
                return "7";
            if (id == 2)
                return "a";
            return "";
        };
        std::vector<uint8_t> okd(3, 0);
        g2.allow_mask(decd, okd.data(), 3);
        CHECK(okd[1] == 1);
        CHECK(okd[2] == 0);
        Gbnf g3;
        CHECK(g3.compile("root ::= \"{\" ws \"a\" ws \"}\"\nws ::= [ \\t]*", e) == Status::Ok);
        CHECK(g3.accept_byte('{'));
        auto dec3 = [](int id) -> std::string {
            if (id == 1)
                return "a";
            if (id == 2)
                return " ";
            if (id == 3)
                return "x";
            return "";
        };
        std::vector<uint8_t> ok3(4, 0);
        g3.allow_mask(dec3, ok3.data(), 4);
        CHECK(ok3[1] == 1 && ok3[2] == 1);
        CHECK(ok3[3] == 0);
        Gbnf bad;
        CHECK(bad.compile("foo ::= \"x\"", e) != Status::Ok);
        CHECK(!bad.ready());
    }
    {
        using namespace mvllm;
        CHECK(kv_common_prefix({1, 2, 3}, {1, 2, 9}) == 2);
        CHECK(kv_common_prefix({1}, {2}) == 0);
        CHECK(conversation_cache_slot({}, 4) == 0);
        CHECK(conversation_cache_slot({{"user", "hi"}}, 1) == 0);
        ChatMessage sys;
        sys.role = "system";
        sys.content = "You are helpful.";
        ChatMessage u1;
        u1.role = "user";
        u1.content = "What is 2+2?";
        ChatMessage u2;
        u2.role = "user";
        u2.content = "Tell me a joke.";
        int s1 = conversation_cache_slot({sys, u1}, 16);
        int s2 = conversation_cache_slot({sys, u2}, 16);
        CHECK(s1 >= 0 && s1 < 16);
        CHECK(s2 >= 0 && s2 < 16);
        ChatMessage asst;
        asst.role = "assistant";
        asst.content = "4";
        CHECK(conversation_cache_slot({sys, u1, asst, u2}, 16) == s1);
        CHECK(conversation_cache_slot({sys, u1}, 16, 3) == 3);
        ChatMessage uimg = u1;
        uimg.image_urls.push_back("data:image/png;base64,QQ==");
        int simg = conversation_cache_slot({sys, uimg}, 16);
        CHECK(simg >= 0 && simg < 16);
        // Extra empty fields must not move a text-only conversation.
        ChatMessage u1b = u1;
        CHECK(conversation_cache_slot({sys, u1b}, 16) == s1);
        ChatMessage sys_tool = sys;
        K3ToolCall tc;
        tc.name = "meteo";
        tc.json = "{\"q\":1}";
        sys_tool.tool_calls.push_back(tc);
        CHECK(conversation_cache_slot({sys_tool, u1}, 4096) !=
              conversation_cache_slot({sys, u1}, 4096));
        CHECK(conversation_cache_slot({sys, uimg}, 4096) !=
              conversation_cache_slot({sys, u1}, 4096));
        SessionStore ss;
        ss.configure(4);
        CHECK(ss.n_slots() == 4);
        CHECK(ss.free_count() == 4);
        CHECK(ss.first_free() == 0);
        CHECK(ss.last_free() == 3);
        CHECK(ss.try_acquire_any() == 0);
        CHECK(ss.busy(0) && ss.first_free() == 1);
        CHECK(ss.last_free() == 3);
        ss.release(0);
        CHECK(ss.valid_slot(0) && ss.valid_slot(3));
        CHECK(!ss.valid_slot(-1) && !ss.valid_slot(4));
        CHECK(ss.try_acquire(1));
        CHECK(ss.busy(1));
        CHECK(ss.first_free() == 0);
        CHECK(ss.free_count() == 3);
        CHECK(!ss.try_acquire(1));
        ss.release(1);
        CHECK(ss.free_count() == 4);
        CHECK(ss.try_acquire(1));
        ss.commit(1, {10, 11, 12});
        CHECK(ss.history_len(1) == 3);
        CHECK(ss.history_total() == 3);
        CHECK(ss.has_history(1));
        CHECK(ss.try_acquire(2));
        ss.release_all();
        CHECK(!ss.busy(1) && !ss.busy(2));
        CHECK(ss.has_history(1));
        CHECK(ss.try_acquire(1));
        CHECK(!ss.has_history(99));
        CHECK(ss.history_len(99) == 0);
        CHECK(ss.match(1, {10, 11, 99}) == 2);
        ss.reset(1);
        CHECK(ss.match(1, {10, 11}) == 0);
        CHECK(!ss.busy(1));
        CHECK(ss.history_total() == 0);
        ss.commit(0, {1, 2});
        ss.commit(2, {3});
        CHECK(ss.history_total() == 3);
        CHECK(ss.try_acquire(2));
        ss.reset_all();
        CHECK(ss.n_slots() == 4);
        CHECK(ss.match(0, {1, 2}) == 0);
        CHECK(ss.match(2, {3}) == 0);
        CHECK(!ss.busy(2));
        CHECK(ss.try_acquire(2));
        CHECK(ss.busy_count() == 1);
        ss.release(2);
        CHECK(ss.busy_count() == 0);
        CHECK(ss.last_free() == 3);
        {
            SessionStore full;
            full.configure(2);
            CHECK(full.last_free() == 1);
            CHECK(full.try_acquire(0) && full.try_acquire(1));
            CHECK(full.last_free() == -1);
            CHECK(full.all_busy());
            CHECK(!full.any_free());
            CHECK(full.try_acquire_last() == -1);
        }
        {
            SessionStore hi;
            hi.configure(4);
            CHECK(hi.last_busy() == -1);
            CHECK(hi.first_busy() == -1);
            CHECK(hi.all_free());
            CHECK(!hi.all_busy());
            CHECK(!hi.any_busy());
            CHECK(hi.any_free());
            CHECK(hi.try_acquire_last() == 3);
            CHECK(hi.busy(3) && hi.last_free() == 2);
            CHECK(hi.last_busy() == 3);
            CHECK(hi.first_busy() == 3);
            CHECK(hi.try_acquire_any() == 0);
            CHECK(hi.try_acquire_last() == 2);
            CHECK(hi.busy(0) && hi.busy(2) && hi.busy(3));
            CHECK(hi.first_free() == 1);
            CHECK(hi.last_busy() == 3);
            CHECK(hi.first_busy() == 0);
            CHECK(!hi.all_free());
            CHECK(hi.any_busy());
            hi.release_all();
            CHECK(hi.all_free());
            CHECK(!hi.any_busy());
        }
        BatchScheduler sch;
        std::string berr;
        CHECK(sch.submit(0, {1}, {}, berr) == 0);
        CHECK(sch.queue_depth() == 0);
        RuntimeConfig rt = runtime_from_env();
        CHECK(rt.kv_slots >= 1 && rt.kv_slots <= 16);
    }
    {
        using namespace mvllm;
        K3Chat1 ch;
        std::string e;
        CHECK(!k3_chat1_parse("hello", ch, e));
        const char *wire = "K3CHAT1\n"
                           "B 1 3 2 1\n"
                           "whyok"
                           "F 11 2\n"
                           "get_weather"
                           "V 4 6 4\n"
                           "citystringRome"
                           "V 4 6 3\n"
                           "daysnumber1e2"
                           "G 1\n";
        CHECK(k3_chat1_parse(wire, ch, e));
        CHECK(ch.think);
        CHECK(ch.msgs.size() == 1);
        CHECK(ch.msgs[0].role == "assistant");
        CHECK(ch.msgs[0].reasoning == "why");
        CHECK(ch.msgs[0].content == "ok");
        CHECK(ch.msgs[0].tool_calls.size() == 1);
        CHECK(ch.msgs[0].tool_calls[0].name == "get_weather");
        CHECK(ch.msgs[0].tool_calls[0].args.size() == 2);
        CHECK(ch.msgs[0].tool_calls[0].args[0].key == "city");
        CHECK(ch.msgs[0].tool_calls[0].args[0].value == "Rome");
        const char *ores = "K3CHAT1\nO 1 11 5\nget_weathersunnyG 0\n";
        K3Chat1 ch2;
        CHECK(k3_chat1_parse(ores, ch2, e));
        CHECK(!ch2.think);
        CHECK(ch2.msgs.size() == 1);
        CHECK(ch2.msgs[0].role == "tool");
        CHECK(ch2.msgs[0].tool_name == "get_weather");
        CHECK(ch2.msgs[0].content == "sunny");
        CHECK(!k3_chat1_parse("K3CHAT1\nO 0 3 2\nfooxxG 0\n", ch2, e));

        std::string ldir = tmpdir();
        write_file(ldir + "/config.json", R"({
          "model_type":"llama","architectures":["LlamaForCausalLM"],
          "hidden_size":32,"num_hidden_layers":1,"vocab_size":16,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
          "intermediate_size":64
        })");
        auto lf = [](int n, int seed) {
            std::vector<float> v(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
                v[static_cast<size_t>(i)] = ((i * 13 + seed) % 11 - 5) * 0.02f;
            return v;
        };
        auto lb = [](const std::vector<float> &v) {
            return std::vector<uint8_t>(reinterpret_cast<const uint8_t *>(v.data()),
                                        reinterpret_cast<const uint8_t *>(v.data() + v.size()));
        };
        using LT = std::tuple<std::string, std::string, std::vector<int64_t>, std::vector<uint8_t>>;
        std::vector<LT> lts;
        auto add = [&](const std::string &n, int o, int i, int seed) {
            auto w = lf(o * i, seed);
            lts.push_back({n, "F32", {o, i}, lb(w)});
        };
        auto add1 = [&](const std::string &n, int o) {
            lts.push_back({n, "F32", {o}, lb(std::vector<float>(static_cast<size_t>(o), 1.f))});
        };
        add("model.embed_tokens.weight", 16, 32, 1);
        add1("model.norm.weight", 32);
        add("lm_head.weight", 16, 32, 2);
        add1("model.layers.0.input_layernorm.weight", 32);
        add1("model.layers.0.post_attention_layernorm.weight", 32);
        add("model.layers.0.self_attn.q_proj.weight", 32, 32, 3);
        add("model.layers.0.self_attn.k_proj.weight", 16, 32, 4);
        add("model.layers.0.self_attn.v_proj.weight", 16, 32, 5);
        add("model.layers.0.self_attn.o_proj.weight", 32, 32, 6);
        add("model.layers.0.mlp.gate_proj.weight", 64, 32, 7);
        add("model.layers.0.mlp.up_proj.weight", 64, 32, 8);
        add("model.layers.0.mlp.down_proj.weight", 32, 64, 9);
        write_safetensors_file(ldir + "/model.safetensors", lts);
        auto le = make_engine(Family::Llama);
        RuntimeConfig lrt;
        std::string lerr;
        CHECK(le->load(ldir, lrt, lerr) == Status::Ok);
        CHECK(le->describe().find("checkpoint=yes") != std::string::npos);
        CHECK(le->config().hidden == 32);
        CHECK(le->config().n_layers == 1);
        GenParams lgp;
        lgp.max_new_tokens = 2;
        lgp.apply_template = false;
        GenResult lout;
        CHECK(le->generate({1, 2}, lgp, lout, lerr) == Status::Ok);
        CHECK(static_cast<int>(lout.tokens.size()) == 2);
        TurnPerf lpf;
        le->turn_perf(lpf, false);
        CHECK(lpf.t_attn > 0.0);
        CHECK(lpf.t_head > 0.0);
        le->turn_perf(lpf, true);
        le->turn_perf(lpf, false);
        CHECK(lpf.t_attn == 0.0 && lpf.t_head == 0.0);
    }
    {
        using namespace mvllm;
        CHECK(stop_cut("abcENDdef", {"END"}) == 3);
        CHECK(stop_cut("hello", {"END"}) == std::string::npos);
        std::string t = "abcENDdef";
        CHECK(trim_stop(t, {"END"}));
        CHECK(t == "abc");
        std::string r, c;
        split_assistant_text(Family::Glm53, "<think>plan</think>Hello", r, c);
        CHECK(r == "plan");
        CHECK(c == "Hello");
        split_assistant_text(Family::Glm53, "<think>still", r, c);
        CHECK(r == "still");
        CHECK(c.empty());
        split_assistant_text(Family::KimiK3,
                             "<|open|>think<|sep|>plan it<|close|>think<|sep|><|open|>response<|sep|>Hi",
                             r, c);
        CHECK(r.find("plan it") != std::string::npos);
        CHECK(c.find("Hi") != std::string::npos);
        split_assistant_text(Family::Llama, "just text", r, c);
        CHECK(r.empty());
        CHECK(c == "just text");
        split_assistant_text(Family::Llama, "<think>plan</think>Hello", r, c);
        CHECK(r == "plan");
        CHECK(c == "Hello");
        split_assistant_text(Family::Llama, "<think>still", r, c);
        CHECK(r == "still");
        CHECK(c.empty());
    }
    {
        using namespace mvllm;
        K3ToolDecl d;
        d.name = "meteo";
        d.description = "weather";
        d.parameters_json = R"({"type":"object"})";
        std::string dec = glm_tool_declare({d});
        CHECK(dec.find("# Tools") != std::string::npos);
        CHECK(dec.find("<tools>") != std::string::npos);
        CHECK(dec.find("meteo") != std::string::npos);
        K3ToolCall call;
        call.name = "meteo";
        call.args.push_back({"citta", "string", "Roma"});
        std::string box = glm_render_tool_calls({call});
        CHECK(box.find("<tool_call>meteo") != std::string::npos);
        CHECK(box.find("<arg_key>citta</arg_key>") != std::string::npos);
        std::string content;
        std::vector<K3ParsedCall> pc;
        CHECK(glm_parse_tool_calls("Let me check." + box, content, pc));
        CHECK(pc.size() == 1 && pc[0].name == "meteo");
        CHECK(pc[0].arguments.find("Roma") != std::string::npos);
        CHECK(content.find("Let me check") != std::string::npos);
        CHECK(content.find("<tool_call>") == std::string::npos);
        Tokenizer gtk;
        std::vector<K3ToolDecl> tds{d};
        std::string gp = gtk.apply_chat(Family::Glm53, {{"user", "hi"}}, false, {}, &tds);
        CHECK(gp.find("[gMASK]<sop>") != std::string::npos);
        CHECK(gp.find("# Tools") != std::string::npos);
        ChatMessage img_user;
        img_user.role = "user";
        img_user.content = "see";
        img_user.image_urls.push_back("a.png");
        img_user.image_urls.push_back("b.png");
        std::string g2 = gtk.apply_chat(Family::Glm53, {img_user}, false);
        const char *ph = "<|begin_of_image|><image><|end_of_image|>";
        size_t p0 = g2.find(ph);
        CHECK(p0 != std::string::npos);
        CHECK(g2.find(ph, p0 + 1) != std::string::npos);
    }
    {
        using namespace mvllm;
        float logits[4] = {1.f, 2.f, 3.f, 4.f};
        const int hist[] = {0, 0, 2};
        apply_penalties(logits, 4, hist, 3, 1.f, 1.f);
        CHECK_NEAR(logits[0], 1.f - 2.f - 1.f, 1e-5);
        CHECK_NEAR(logits[1], 2.f, 1e-5);
        CHECK_NEAR(logits[2], 3.f - 1.f - 1.f, 1e-5);
        CHECK_NEAR(logits[3], 4.f, 1e-5);
        float ident[3] = {0.5f, 1.5f, -0.25f};
        const int nohist[] = {1};
        apply_penalties(ident, 3, nohist, 1, 0.f, 0.f);
        CHECK_NEAR(ident[0], 0.5f, 1e-6);
        CHECK_NEAR(ident[1], 1.5f, 1e-6);
        const std::pair<int, float> bias[] = {{1, 2.5f}, {9, 99.f}, {-1, 1.f}};
        apply_logit_bias(ident, 3, bias, 3);
        CHECK_NEAR(ident[1], 4.f, 1e-5);
        CHECK_NEAR(ident[0], 0.5f, 1e-6);
        float even[3] = {0.f, 0.f, 0.f};
        CHECK_NEAR(token_logprob(even, 3, 0), -std::log(3.f), 1e-5);
        uint8_t allow[3] = {1, 0, 1};
        CHECK_NEAR(token_logprob(even, 3, 0, allow), -std::log(2.f), 1e-5);
        CHECK(token_logprob(even, 3, 1, allow) < -1e20f);
        GenLogprob tops[4];
        int ntop = 0;
        float ranked[4] = {1.f, 4.f, 2.f, 3.f};
        top_logprobs(ranked, 4, 2, tops, &ntop, nullptr);
        CHECK(ntop == 2 && tops[0].token == 1 && tops[1].token == 3);
        CHECK(tops[0].logprob > tops[1].logprob);
        float greedy[3] = {0.f, 5.f, 1.f};
        float lp = 0.f;
        int tok = sample_token(greedy, 3, 0.f, 1.f, nullptr, nullptr, &lp);
        CHECK(tok == 1);
        CHECK_NEAR(lp, token_logprob(greedy, 3, 1), 1e-5);
        std::vector<std::pair<int, float>> lb;
        CHECK(extract_json_logit_bias("{\"logit_bias\":{\"7\":1.5,\"11\":-2}}", lb));
        CHECK(lb.size() == 2);
        CHECK(lb[0].first == 7 && std::fabs(lb[0].second - 1.5f) < 1e-5f);
        CHECK(lb[1].first == 11 && std::fabs(lb[1].second + 2.f) < 1e-5f);
        float fp = 0.f, pp = 0.f;
        CHECK(extract_json_number("{\"frequency_penalty\":0.4,\"presence_penalty\":0.2}",
                                  "frequency_penalty", fp) &&
              std::fabs(fp - 0.4f) < 1e-5f);
        CHECK(extract_json_number("{\"frequency_penalty\":0.4,\"presence_penalty\":0.2}",
                                  "presence_penalty", pp) &&
              std::fabs(pp - 0.2f) < 1e-5f);
        int mct = 0, tlp = 0;
        CHECK(extract_json_int("{\"max_completion_tokens\":17}", "max_completion_tokens", mct) &&
              mct == 17);
        bool lpb = false;
        CHECK(extract_json_bool("{\"logprobs\":true,\"top_logprobs\":5}", "logprobs", lpb) && lpb);
        CHECK(extract_json_int("{\"logprobs\":true,\"top_logprobs\":5}", "top_logprobs", tlp) &&
              tlp == 5);
        CHECK(stop_cut("hi<|user|>next", {"<|user|>", "<|observation|>"}) == 2);
        CHECK(stop_cut("tool<|observation|>x", {"<|user|>", "<|observation|>"}) == 4);
    }
    {
        using namespace mvllm;
        std::vector<ChatMessage> msgs;
        GenParams gp;
        std::string err;
        CHECK(!anthropic_to_chat("{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", msgs,
                                 gp, err));
        CHECK(err.find("max_tokens") != std::string::npos);
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":16,\"system\":\"be brief\",\"thinking\":{\"type\":\"enabled\"},"
            "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            "\"tools\":[{\"name\":\"meteo\",\"description\":\"w\","
            "\"input_schema\":{\"type\":\"object\"}}],"
            "\"tool_choice\":{\"type\":\"tool\",\"name\":\"meteo\"}}",
            msgs, gp, err));
        CHECK(gp.max_new_tokens == 16);
        CHECK(gp.think);
        CHECK(gp.tool_choice == "meteo");
        CHECK(gp.tools.size() == 1 && gp.tools[0].name == "meteo");
        CHECK(msgs.size() == 2 && msgs[0].role == "system" && msgs[0].content == "be brief");
        CHECK(msgs[1].role == "user" && msgs[1].content == "hi");
        GenResult gr;
        gr.text = "ciao";
        gr.prompt_tokens = 3;
        gr.completion_tokens = 1;
        std::string resp = anthropic_messages_response("msg_1", "kimi", gr);
        CHECK(resp.find("\"type\":\"message\"") != std::string::npos);
        CHECK(resp.find("ciao") != std::string::npos);
        CHECK(resp.find("\"stop_reason\"") != std::string::npos);
        CHECK(resp.find("\"input_tokens\":3") != std::string::npos);
        std::string start = anthropic_sse_start("msg_1", "kimi");
        CHECK(start.find("event: message_start") != std::string::npos);
        std::string delta = anthropic_sse_delta("x");
        CHECK(delta.find("event: content_block_delta") != std::string::npos);
        CHECK(delta.find("text_delta") != std::string::npos);
        std::string stop = anthropic_sse_stop("end_turn");
        CHECK(stop.find("event: message_delta") != std::string::npos);
        CHECK(stop.find("event: message_stop") != std::string::npos);
        std::string mid = openai_models_response("glm53");
        CHECK(mid.find("glm53") != std::string::npos);
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            "\"stop_sequences\":[\"END\",\"\"]}",
            msgs, gp, err));
        CHECK(gp.stop.size() == 1 && gp.stop[0] == "END");
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":["
            "{\"type\":\"text\",\"text\":\"see \"},"
            "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\","
            "\"data\":\"QQ==\"}}]}]}",
            msgs, gp, err));
        CHECK(msgs.size() == 1 && msgs[0].content == "see ");
        CHECK(msgs[0].image_urls.size() == 1);
        CHECK(msgs[0].image_urls[0] == "data:image/png;base64,QQ==");
        std::string bstart = anthropic_sse_block_start(0);
        CHECK(bstart.find("event: content_block_start") != std::string::npos);
        CHECK(bstart.find("content_block") != std::string::npos);
        std::string bstop = anthropic_sse_block_stop(0);
        CHECK(bstop.find("event: content_block_stop") != std::string::npos);
    }
    {
        using namespace mvllm;
        std::string e;
        Gbnf gj;
        CHECK(gj.compile(json_object_gbnf(), e) == Status::Ok);
        CHECK(gj.ready());
        CHECK(gj.accept_bytes("{\"a\":1}"));
        std::string serr;
        CHECK(json_schema_to_gbnf("", serr).empty());
        CHECK(!serr.empty());
        std::string sch = json_schema_to_gbnf(
            "{\"type\":\"object\",\"properties\":{\"ok\":{\"type\":\"boolean\"}},"
            "\"required\":[\"ok\"],\"additionalProperties\":false}",
            serr);
        CHECK(!sch.empty());
        Gbnf gs;
        CHECK(gs.compile(sch, e) == Status::Ok);
        CHECK(gs.accept_bytes("{\"ok\":true}"));
        std::string gram, rerr;
        CHECK(extract_response_format("{\"response_format\":{\"type\":\"json_object\"}}", gram,
                                      rerr));
        CHECK(!gram.empty());
        std::string keep = "root ::= \"x\"";
        CHECK(extract_response_format("{\"response_format\":{\"type\":\"json_object\"}}", keep,
                                      rerr));
        CHECK(keep == "root ::= \"x\"");
        std::string jsgram;
        CHECK(extract_response_format(
            "{\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{"
            "\"name\":\"t\",\"schema\":{\"type\":\"object\",\"properties\":{"
            "\"ok\":{\"type\":\"boolean\"}},\"required\":[\"ok\"],"
            "\"additionalProperties\":false}}}}",
            jsgram, rerr));
        CHECK(!jsgram.empty());
        std::string cached =
            openai_chat_response("id1", "kimi", "hello", 3, 2, {}, "stop", 4);
        CHECK(cached.find("\"cached_tokens\":4") != std::string::npos);
        CHECK(cached.find("prompt_tokens_details") != std::string::npos);
        std::string nocache = openai_chat_response("id1", "kimi", "hello", 3, 2);
        CHECK(nocache.find("cached_tokens") == std::string::npos);
    }
    {
        using namespace mvllm;
        float rp[4] = {2.f, -1.f, 0.5f, -0.5f};
        const int hist[] = {0, 0, 1, 3};
        apply_repetition_penalty(rp, 4, hist, 4, 1.2f);
        CHECK_NEAR(rp[0], 2.f / 1.2f, 1e-5);
        CHECK_NEAR(rp[1], -1.2f, 1e-5);
        CHECK_NEAR(rp[2], 0.5f, 1e-5);
        CHECK_NEAR(rp[3], -0.6f, 1e-5);
        float ident[3] = {1.f, 2.f, 3.f};
        apply_repetition_penalty(ident, 3, hist, 4, 1.f);
        CHECK_NEAR(ident[0], 1.f, 1e-6);
        apply_top_k(ident, 3, 0);
        CHECK_NEAR(ident[2], 3.f, 1e-6);
        float tk[4] = {1.f, 3.f, 3.f, 2.f};
        apply_top_k(tk, 4, 1);
        CHECK(tk[1] == 3.f);
        CHECK(tk[0] < -1e20f && tk[2] < -1e20f && tk[3] < -1e20f);
        float mp[4] = {0.f, 1.f, 3.f, -1.f};
        apply_min_p(mp, 4, 0.1f);
        CHECK(mp[1] == 1.f && mp[2] == 3.f);
        CHECK(mp[0] < -1e20f && mp[3] < -1e20f);
        GenParams gp;
        std::string merr;
        CHECK(mux_apply_extra_json("", gp, merr));
        CHECK(mux_apply_extra_json(
            "{\"stop\":[\"END\"],\"top_k\":8,\"repetition_penalty\":1.1,\"min_p\":0.05}", gp,
            merr));
        CHECK(gp.stop.size() == 1 && gp.stop[0] == "END");
        CHECK(gp.top_k == 8);
        CHECK_NEAR(gp.repetition_penalty, 1.1f, 1e-5);
        CHECK_NEAR(gp.min_p, 0.05f, 1e-5);
        CHECK(!mux_apply_extra_json("not-json", gp, merr));
        uint8_t raw[12] = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
        std::vector<float> rgb;
        int iw = 0, ih = 0;
        std::string ierr;
        CHECK(mux_decode_image(raw, 12, 2, 2, rgb, iw, ih, ierr) == Status::Ok);
        CHECK(iw == 2 && ih == 2 && rgb.size() == 12);
        CHECK(rgb[0] > 0.9f && rgb[1] < 0.1f);
        uint8_t f32le[8] = {0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0};
        CHECK(mux_decode_image(f32le, 8, 1, 2, rgb, iw, ih, ierr) == Status::Ok);
        CHECK(iw == 2 && ih == 1 && rgb.size() == 2);
        CHECK_NEAR(rgb[0], 1.f, 1e-6);
        CHECK_NEAR(rgb[1], -2.f, 1e-6);
        const char *old_key = std::getenv("MVLLM_API_KEY");
        std::string old_key_s = old_key ? old_key : "";
        unsetenv("MVLLM_API_KEY");
        CHECK(api_key_ok("nope"));
        setenv("MVLLM_API_KEY", "secret", 1);
        CHECK(api_key_ok("Bearer secret"));
        CHECK(api_key_ok("", "secret"));
        CHECK(!api_key_ok("Bearer other"));
        CHECK(!api_key_ok("", "other"));
        if (!old_key_s.empty())
            setenv("MVLLM_API_KEY", old_key_s.c_str(), 1);
        else
            unsetenv("MVLLM_API_KEY");
        int topk = 0;
        float minp = 0.f, rpen = 0.f;
        CHECK(extract_json_int("{\"top_k\":16}", "top_k", topk) && topk == 16);
        CHECK(extract_json_number("{\"min_p\":0.05}", "min_p", minp) &&
              std::fabs(minp - 0.05f) < 1e-5f);
        CHECK(extract_json_number("{\"repetition_penalty\":1.15}", "repetition_penalty", rpen) &&
              std::fabs(rpen - 1.15f) < 1e-5f);
        bool echo = false;
        CHECK(extract_json_bool("{\"echo\":true}", "echo", echo) && echo);
        int nchoice = 0;
        CHECK(extract_json_int("{\"n\":2}", "n", nchoice) && nchoice == 2);
        std::string fp = openai_chat_response("id1", "kimi", "hello", 3, 2);
        CHECK(fp.find("\"system_fingerprint\":\"fp_mvllm\"") != std::string::npos);
        int hseed = 0, hlays = 0;
        CHECK(extract_json_int("{\"seed\":9,\"dit_layers\":12}", "seed", hseed) && hseed == 9);
        CHECK(extract_json_int("{\"seed\":9,\"dit_layers\":12}", "dit_layers", hlays) &&
              hlays == 12);
        std::vector<ChatMessage> amsgs;
        GenParams agp;
        std::string aerr;
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":8,\"metadata\":{\"user_id\":\"u\"},\"top_k\":5,\"seed\":7,"
            "\"min_p\":0.1,\"repetition_penalty\":1.1,"
            "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\","
            "\"cache_control\":{\"type\":\"ephemeral\"}}]}]}",
            amsgs, agp, aerr));
        CHECK(agp.top_k == 5 && agp.seed == 7);
        CHECK_NEAR(agp.min_p, 0.1f, 1e-5);
        CHECK_NEAR(agp.repetition_penalty, 1.1f, 1e-5);
        CHECK(amsgs.size() == 1 && amsgs[0].content == "hi");
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":8,\"thinking\":{\"type\":\"disabled\",\"budget_tokens\":99999},"
            "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
            amsgs, agp, aerr));
        CHECK(!agp.think);
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":8,\"thinking\":{\"budget_tokens\":8000},"
            "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}",
            amsgs, agp, aerr));
        CHECK(agp.think && agp.reasoning_effort == "medium");
        std::string fc;
        CHECK(extract_tool_choice("{\"function_call\":\"none\"}", fc) && fc == "none");
        CHECK(extract_tool_choice("{\"function_call\":{\"name\":\"meteo\"}}", fc) &&
              fc == "meteo");
        std::string suf;
        CHECK(extract_json_string("{\"suffix\":\"TAIL\"}", "suffix", suf) && suf == "TAIL");
        {
            char a0[] = "micro-vllm";
            char a1[] = "generate";
            char a2[] = "--stop";
            char a3[] = "END";
            char a4[] = "--json";
            char a5[] = "--top-k";
            char a6[] = "8";
            char a7[] = "--min-p";
            char a8[] = "0.05";
            char *av[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
            GenParams cgp;
            CliGenExtras cex;
            std::string cerr;
            CHECK(apply_cli_gen_flags(9, av, cgp, cex, cerr));
            CHECK(cgp.stop.size() == 1 && cgp.stop[0] == "END");
            CHECK(cgp.top_k == 8);
            CHECK_NEAR(cgp.min_p, 0.05f, 1e-5);
            CHECK(!cgp.grammar.empty());
            char nj0[] = "micro-vllm";
            char nj1[] = "--json";
            char nj2[] = "--no-json";
            char *njv[] = {nj0, nj1, nj2};
            CHECK(apply_cli_gen_flags(3, njv, cgp, cex, cerr));
            CHECK(cgp.grammar.empty());
            char ng0[] = "micro-vllm";
            char ng1[] = "--no-grammar";
            char ng2[] = "--json";
            char *ngv[] = {ng0, ng1, ng2};
            CHECK(apply_cli_gen_flags(3, ngv, cgp, cex, cerr));
            CHECK(!cgp.grammar.empty());
        }
        {
            char b0[] = "micro-vllm";
            char b1[] = "--logit-bias";
            char b2[] = "7:1.5";
            char b3[] = "--logit-bias";
            char b4[] = "11:-2";
            char *bv[] = {b0, b1, b2, b3, b4};
            GenParams cgp;
            CliGenExtras cex;
            std::string cerr;
            CHECK(apply_cli_gen_flags(5, bv, cgp, cex, cerr));
            CHECK(cgp.logit_bias.size() == 2);
            CHECK(cgp.logit_bias[0].first == 7);
            CHECK_NEAR(cgp.logit_bias[0].second, 1.5f, 1e-5);
            CHECK(cgp.logit_bias[1].first == 11);
            CHECK_NEAR(cgp.logit_bias[1].second, -2.f, 1e-5);
            char c0[] = "micro-vllm";
            char c1[] = "--logit-bias";
            char c2[] = "nocolon";
            char *cv[] = {c0, c1, c2};
            CHECK(!apply_cli_gen_flags(3, cv, cgp, cex, cerr));
            char n0[] = "micro-vllm";
            char n1[] = "--max-tokens";
            char n2[] = "32";
            char *nv[] = {n0, n1, n2};
            CHECK(apply_cli_gen_flags(3, nv, cgp, cex, cerr));
            CHECK(cgp.max_new_tokens == 32);
            char z0[] = "micro-vllm";
            char z1[] = "--n";
            char z2[] = "0";
            char *zv[] = {z0, z1, z2};
            CHECK(!apply_cli_gen_flags(3, zv, cgp, cex, cerr));
            char l0[] = "micro-vllm";
            char l1[] = "--logprobs";
            char l2[] = "5";
            char *lv[] = {l0, l1, l2};
            CHECK(apply_cli_gen_flags(3, lv, cgp, cex, cerr));
            CHECK(cgp.logprobs == 5);
            char nl0[] = "micro-vllm";
            char nl1[] = "--no-logprobs";
            char *nlv[] = {nl0, nl1};
            CHECK(apply_cli_gen_flags(2, nlv, cgp, cex, cerr));
            CHECK(cgp.logprobs == 0);
            char nl2[] = "micro-vllm";
            char nl3[] = "--no-logprobs";
            char nl4[] = "--logprobs";
            char nl5[] = "3";
            char *nlw[] = {nl2, nl3, nl4, nl5};
            CHECK(apply_cli_gen_flags(4, nlw, cgp, cex, cerr));
            CHECK(cgp.logprobs == 3);
            char p0[] = "micro-vllm";
            char p1[] = "--top-logprobs";
            char p2[] = "-1";
            char *pv[] = {p0, p1, p2};
            CHECK(!apply_cli_gen_flags(3, pv, cgp, cex, cerr));
            char s0[] = "micro-vllm";
            char s1[] = "--cache-slot";
            char s2[] = "3";
            char *sv[] = {s0, s1, s2};
            CHECK(apply_cli_gen_flags(3, sv, cgp, cex, cerr));
            CHECK(cgp.cache_slot == 3);
            char t0[] = "micro-vllm";
            char t1[] = "--slot";
            char t2[] = "-1";
            char *tv[] = {t0, t1, t2};
            CHECK(!apply_cli_gen_flags(3, tv, cgp, cex, cerr));
            char e0[] = "micro-vllm";
            char e1[] = "--reasoning-effort";
            char e2[] = "medium";
            char *ev[] = {e0, e1, e2};
            CHECK(apply_cli_gen_flags(3, ev, cgp, cex, cerr));
            CHECK(cgp.reasoning_effort == "medium" && cgp.think);
            char u0[] = "micro-vllm";
            char u1[] = "--reasoning-effort";
            char u2[] = "none";
            char *uv[] = {u0, u1, u2};
            CHECK(apply_cli_gen_flags(3, uv, cgp, cex, cerr));
            CHECK(cgp.reasoning_effort == "none" && !cgp.think);
            char w0[] = "micro-vllm";
            char w1[] = "--reasoning-effort";
            char w2[] = "max";
            char *wv[] = {w0, w1, w2};
            CHECK(!apply_cli_gen_flags(3, wv, cgp, cex, cerr));
            char x0[] = "micro-vllm";
            char x1[] = "--tool-choice";
            char x2[] = "none";
            char *xv[] = {x0, x1, x2};
            CHECK(apply_cli_gen_flags(3, xv, cgp, cex, cerr));
            CHECK(cgp.tool_choice == "none");
            char y0[] = "micro-vllm";
            char y1[] = "--tool-choice";
            char y2[] = "meteo";
            char *yv[] = {y0, y1, y2};
            CHECK(apply_cli_gen_flags(3, yv, cgp, cex, cerr));
            CHECK(cgp.tool_choice == "meteo");
            char i0[] = "micro-vllm";
            char i1[] = "--function-call";
            char i2[] = "none";
            char *iv[] = {i0, i1, i2};
            CHECK(apply_cli_gen_flags(3, iv, cgp, cex, cerr));
            CHECK(cgp.tool_choice == "none");
            char r0[] = "micro-vllm";
            char r1[] = "--chat";
            char r2[] = "--raw";
            char *rv[] = {r0, r1, r2};
            CHECK(apply_cli_gen_flags(3, rv, cgp, cex, cerr));
            CHECK(!cgp.apply_template);
            char q0[] = "micro-vllm";
            char q1[] = "--no-chat";
            char q2[] = "--chat";
            char *qv[] = {q0, q1, q2};
            CHECK(apply_cli_gen_flags(3, qv, cgp, cex, cerr));
            CHECK(cgp.apply_template);
            std::string tdir = tmpdir();
            write_file(tdir + "/tools.json",
                       R"({"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]})");
            std::string tpath = tdir + "/tools.json";
            char f0[] = "micro-vllm";
            char f1[] = "--tools";
            std::vector<char> f2(tpath.begin(), tpath.end());
            f2.push_back(0);
            char *fv[] = {f0, f1, f2.data()};
            CHECK(apply_cli_gen_flags(3, fv, cgp, cex, cerr));
            CHECK(cgp.tools.size() == 1 && cgp.tools[0].name == "get_weather");
            char nt0[] = "micro-vllm";
            char nt1[] = "--no-tools";
            char *ntv[] = {nt0, nt1};
            CHECK(apply_cli_gen_flags(2, ntv, cgp, cex, cerr));
            CHECK(cgp.tools.empty());
            char nt2[] = "micro-vllm";
            char nt3[] = "--no-tools";
            char nt4[] = "--tools";
            std::vector<char> nt5(tpath.begin(), tpath.end());
            nt5.push_back(0);
            char *ntw[] = {nt2, nt3, nt4, nt5.data()};
            CHECK(apply_cli_gen_flags(4, ntw, cgp, cex, cerr));
            CHECK(cgp.tools.size() == 1 && cgp.tools[0].name == "get_weather");
            char g0[] = "micro-vllm";
            char g1[] = "--tools";
            char g2[] = "/no/such/tools.json";
            char *gv[] = {g0, g1, g2};
            CHECK(!apply_cli_gen_flags(3, gv, cgp, cex, cerr));
        }
        std::string done = mux_format_done(7, 3, 10, 0, 2);
        CHECK(done.find("DONE 7 STAT 3") != std::string::npos);
        CHECK(done.find("10 0 2") != std::string::npos);
        CHECK(mux_format_done(7, 3, 10, 0, 2, 12.5, 80.0, 1.25).find("12.50 80.0 1.25 10 0 2") !=
              std::string::npos);
        std::string rtoks =
            openai_chat_response("id1", "kimi", "hello", 3, 2, "plan", "stop", 0, 4);
        CHECK(rtoks.find("\"reasoning_tokens\":4") != std::string::npos);
        CHECK(rtoks.find("completion_tokens_details") != std::string::npos);
        std::string nor =
            openai_chat_response("id1", "kimi", "hello", 3, 2);
        CHECK(nor.find("reasoning_tokens") == std::string::npos);
        std::string hz = health_json(nullptr);
        CHECK(hz.find("\"ok\":true") != std::string::npos);
        CHECK(hz.find("rss_gb") != std::string::npos);
        CHECK(hz.find("kv_slots") != std::string::npos);
        CHECK(hz.find("\"jobs\"") == std::string::npos);
        CHECK(hz.find("\"live\"") == std::string::npos);
        CHECK(hz.find("\"model\"") == std::string::npos);
        CHECK(hz.find("\"created\":" + std::to_string(serve_created())) != std::string::npos);
        CHECK(hz.find("\"cores\":") != std::string::npos);
        CHECK(hz.find("\"ram_gb\":") != std::string::npos);
        CHECK(hz.find("\"ngpu\":") != std::string::npos);
        CHECK(hz.find("\"busy_slots\":0") != std::string::npos);
        CHECK(hz.find("\"hist_tokens\":0") != std::string::npos);
        CHECK(hz.find("\"free_slots\":0") != std::string::npos);
        CHECK(hz.find("\"idle\"") == std::string::npos);
        CHECK(hz.find("\"failed\"") == std::string::npos);
        HwInfo hjs{};
        hjs.cores = 8;
        hjs.ram_total_gb = 16;
        hjs.ram_avail_gb = 8;
        hjs.ngpu = 0;
        hjs.cpu = "test\"cpu";
        CHECK(hwinfo_json(hjs) ==
              "{\"cores\":8,\"ram_total_gb\":16.00,\"ram_avail_gb\":8.00,\"ngpu\":0,"
              "\"vram_total_gb\":0.00,\"cpu\":\"test\\\"cpu\",\"gpu\":\"none\"}");
        CHECK(hwinfo_line(hjs) == "HWINFO 8 16.00 8.00 0 0.00 test\"cpu|\n");
        std::string mo = openai_model_object("glm53");
        CHECK(serve_created() >= 0);
        CHECK(mo.find("\"created\":" + std::to_string(serve_created())) != std::string::npos);
        CHECK(mo.find("glm53") != std::string::npos);
        CHECK(openai_model_object("glm53", 0).find("\"created\":0") != std::string::npos);
        CHECK(openai_models_response("glm53").find("\"created\":" +
                                                   std::to_string(serve_created())) !=
              std::string::npos);
        {
            char v0[] = "micro-vllm";
            char v1[] = "video";
            char v2[] = "--width";
            char v3[] = "1280";
            char v4[] = "--seed";
            char v5[] = "7";
            char v6[] = "--audio";
            char v7[] = "a.wav";
            char *vv[] = {v0, v1, v2, v3, v4, v5, v6, v7};
            H3GenParams hp;
            std::string verr;
            CHECK(apply_cli_video_flags(8, vv, hp, verr));
            CHECK(hp.width == 1280);
            CHECK(hp.seed == 7);
            CHECK(hp.audio_path == "a.wav");
        }
        Tokenizer ltk;
        ChatMessage lu;
        lu.role = "user";
        lu.content = "see";
        lu.image_urls.push_back("x.png");
        lu.image_urls.push_back("y.png");
        std::string lp = ltk.apply_chat(Family::Llama, {lu}, false);
        CHECK(lp.find("see<image><image>") != std::string::npos);
        std::string mj = metrics_json(12, 384, 4, 0);
        CHECK(mj.find("\"requests\":12") != std::string::npos);
        CHECK(mj.find("\"tokens_out\":384") != std::string::npos);
        CHECK(mj.find("\"kv_slots\":4") != std::string::npos);
        CHECK(mj.find("\"queue\":0") != std::string::npos);
        CHECK(mj.find("\"created\":" + std::to_string(serve_created())) != std::string::npos);
        CHECK(mj.find("\"busy_slots\":0") != std::string::npos);
        CHECK(mj.find("\"hist_tokens\":0") != std::string::npos);
        CHECK(mj.find("\"free_slots\":0") != std::string::npos);
        CHECK(mj.find("\"failed\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 3).find("\"busy_slots\":3") !=
              std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2).find("\"hist_tokens\":5") !=
              std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2).find("\"free_slots\":2") !=
              std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4).find("\"failed\":4") !=
              std::string::npos);
        CHECK(mj.find("\"idle\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7).find("\"idle\":7") !=
              std::string::npos);
        CHECK(mj.find("\"jobs\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9).find("\"jobs\":9") !=
              std::string::npos);
        CHECK(mj.find("\"live\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6).find("\"live\":6") !=
              std::string::npos);
        CHECK(mj.find("\"capacity\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8).find("\"capacity\":8") !=
              std::string::npos);
        CHECK(mj.find("\"admitted\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8, 11).find(
                  "\"admitted\":11") != std::string::npos);
        CHECK(mj.find("\"completed\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8, 11, 13).find(
                  "\"completed\":13") != std::string::npos);
        CHECK(mj.find("\"rejected\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8, 11, 13, 2).find(
                  "\"rejected\":2") != std::string::npos);
        CHECK(mj.find("\"timed_out\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8, 11, 13, 2, 4).find(
                  "\"timed_out\":4") != std::string::npos);
        CHECK(mj.find("\"cancelled\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8, 11, 13, 2, 4, 1).find(
                  "\"cancelled\":1") != std::string::npos);
        CHECK(mj.find("\"queue_timeout\":0") != std::string::npos);
        CHECK(metrics_json(1, 2, 4, 0, 0, 0, 0, 0, 0, 5, 2, 4, 7, 9, 6, 8, 11, 13, 2, 4, 1, 300)
                  .find("\"queue_timeout\":300") != std::string::npos);
        std::string st = mux_format_stat(2);
        CHECK(st.find("STAT 2 0.00 0.0 0.00") != std::string::npos);
        std::string td = anthropic_sse_thinking_delta("plan");
        CHECK(td.find("event: content_block_delta") != std::string::npos);
        CHECK(td.find("thinking_delta") != std::string::npos);
        std::string ping = anthropic_sse_ping();
        CHECK(ping.find("event: ping") != std::string::npos);
        CHECK(ping.find("\"type\":\"ping\"") != std::string::npos);
        std::string aserr = anthropic_sse_error(nullptr, "generate failed");
        CHECK(aserr.find("event: error") != std::string::npos);
        CHECK(aserr.find("\"type\":\"error\"") != std::string::npos);
        CHECK(aserr.find("\"api_error\"") != std::string::npos);
        CHECK(aserr.find("generate failed") != std::string::npos);
        CHECK(anthropic_sse_error("", "x").find("\"api_error\"") != std::string::npos);
        CHECK(anthropic_sse_error("overloaded_error", "busy").find("overloaded_error") !=
              std::string::npos);
        std::string ka = openai_sse_keepalive("id1", "kimi", true, false);
        CHECK(ka.find("reasoning_content") != std::string::npos);
        CHECK(openai_sse_keepalive("id1", "kimi", true, true).find("\".\"") != std::string::npos);
        CHECK(openai_sse_keepalive("id1", "kimi", false, false).find("text_completion") !=
              std::string::npos);
        CHECK(td.find("plan") != std::string::npos);
        std::string th = anthropic_sse_block_start(0, "thinking");
        CHECK(th.find("event: content_block_start") != std::string::npos);
        CHECK(th.find("thinking") != std::string::npos);
        std::string tx = anthropic_sse_block_start(0);
        CHECK(tx.find("\"type\":\"text\"") != std::string::npos);
        std::string srv = openai_chat_response("id1", "kimi", "hello", 3, 2);
        CHECK(srv.find("\"service_tier\":\"default\"") != std::string::npos);
        std::string usr;
        CHECK(extract_json_string("{\"user\":\"alice\"}", "user", usr) && usr == "alice");
        std::string tail = "Hello\n<|im_end|>\n<|assistant|> </s>  ";
        trim_assistant_tail(tail);
        CHECK(tail == "Hello");
        std::string tail2 = "Hi<|endoftext|>\n<|eom_id|>  ";
        trim_assistant_tail(tail2);
        CHECK(tail2 == "Hi");
        std::string tail3 = "Yo <|end_of_text|>\n  ";
        trim_assistant_tail(tail3);
        CHECK(tail3 == "Yo");
        std::string tail4 = "Ok <|end|>\n  ";
        trim_assistant_tail(tail4);
        CHECK(tail4 == "Ok");
        std::string tail5 = "Go <|endofprompt|>\n  ";
        trim_assistant_tail(tail5);
        CHECK(tail5 == "Go");
        std::string tail6 = "Bye <|end_of_turn|>\n  ";
        trim_assistant_tail(tail6);
        CHECK(tail6 == "Bye");
        std::string r, c;
        split_assistant_text(Family::Llama, "<think>plan</think>Hello<|eot_id|>\n", r, c);
        CHECK(r == "plan");
        CHECK(c == "Hello");
        BatchScheduler sch;
        sch.configure(4, 10);
        CHECK(sch.max_queue() == 4);
        CHECK(sch.queue_timeout_s() == 10);
        SchedulerSnapshot snap{};
        sch.snapshot(snap);
        CHECK(snap.max_queue == 4 && snap.queue_timeout_seconds == 10);
        CHECK(snap.admitted == 0 && snap.rejected == 0);
        CHECK(snap.busy_slots == 0);
        CHECK(snap.jobs == 0);
        CHECK(snap.live == 0);
        CHECK(snap.idle == 1);
        CHECK(snap.hist_tokens == 0);
        CHECK(snap.failed == 0);
        CHECK(sch.failed_count() == 0);
        CHECK(sch.live_count() == 0);
        CHECK(sch.idle_count() == 1);
        CHECK(sch.capacity() == 1);
        CHECK(sch.admitted_count() == 0);
        CHECK(sch.completed_count() == 0);
        CHECK(sch.rejected_count() == 0);
        CHECK(sch.timed_out_count() == 0);
        CHECK(sch.cancelled_count() == 0);
        CHECK(queue_error_json("queue_full").find("\"code\":\"queue_full\"") != std::string::npos);
        CHECK(queue_error_json("queue_timeout").find("queue_timeout") != std::string::npos);
        CHECK(queue_wait_header(0.0123).find("x-colibri-queue-wait-ms: 12") != std::string::npos);
        CHECK(host_header_name("LocalHost:8000") == "localhost");
        CHECK(host_header_name("[::1]:8000") == "::1");
        CHECK(host_allowed("evil.example", "127.0.0.1", "") == false);
        CHECK(host_allowed("127.0.0.1:8000", "0.0.0.0", ""));
        CHECK(host_allowed("proxy.local", "127.0.0.1", "proxy.local"));
        CHECK(host_allowed("any.example", "127.0.0.1", "*"));
        CHECK(io::mime_type("x.html") == "text/html; charset=utf-8");
        CHECK(io::url_unquote("/a%2eb") == "/a.b");
        {
            std::string root = tmpdir();
            write_file(root + "/index.html", "<html>ok</html>");
            write_file(root + "/app.js", "1");
            std::string p, ct;
            CHECK(io::static_resolve(root, "/", p, ct) && ct.find("text/html") != std::string::npos);
            CHECK(io::static_resolve(root, "/app.js", p, ct) && ct == "application/javascript");
            CHECK(!io::static_resolve(root, "/../etc/passwd", p, ct));
            CHECK(!io::static_resolve(root, "/missing.css", p, ct));
        }
        CHECK(sch.running_count() == 0);
        CHECK(sch.queued_count() == 0);
        CHECK(sch.n_jobs() == 0);
        CHECK(sch.queue_depth() == 0);
        std::string mj2 = metrics_json(1, 2, 4, 0, 1, 3, 8);
        CHECK(mj2.find("\"running\":1") != std::string::npos);
        CHECK(mj2.find("\"queued\":3") != std::string::npos);
        CHECK(mj2.find("\"max_queue\":8") != std::string::npos);
        CHECK(health_json(nullptr).find("running") == std::string::npos);
        ShardReport sr;
        sr.family = Family::KimiK3;
        sr.synth = true;
        sr.shards_ok = true;
        sr.note = "synthetic";
        std::string sj = format_shard_report_json(sr);
        CHECK(sj.find("\"family\":\"kimi_k3\"") != std::string::npos);
        CHECK(sj.find("\"synth\":true") != std::string::npos);
        CHECK(sj.find("\"shards_ok\":true") != std::string::npos);
        std::string su = anthropic_sse_stop("end_turn", nullptr, 5, 3);
        CHECK(su.find("\"input_tokens\":3") != std::string::npos);
        CHECK(su.find("\"output_tokens\":5") != std::string::npos);
        CHECK(su.find("event: message_delta") != std::string::npos);
    }
    {
        using namespace mvllm;
        int aw = 0, ah = 0;
        CHECK(h3_adapt_canvas(1920, 1080, &aw, &ah) && aw == 1344 && ah == 768);
        CHECK(h3_adapt_canvas(1080, 1920, &aw, &ah) && aw == 768 && ah == 1344);
        CHECK(h3_adapt_canvas(32, 32, &aw, &ah) && aw == 768 && ah == 768);
        CHECK(!h3_adapt_canvas(0, 10, &aw, &ah));
        CHECK(h3_reference_image_canvas(64, 64, 864, 480, 0, &aw, &ah) && aw == 64 && ah == 64);
        CHECK(h3_reference_image_canvas(1920, 1080, 512, 512, 0, &aw, &ah) && aw == 672 &&
              ah == 384);
        CHECK(h3_reference_video_canvas(1920, 1080, &aw, &ah) && aw == 1344 && ah == 768);
        CHECK(h3_reference_video_canvas(640, 360, &aw, &ah) && aw == 640 && ah == 352);
        CHECK(!h3_reference_image_canvas(64, 64, 864, 480, -1, &aw, &ah));

        H3Rng rng;
        h3_rng_seed(rng, 1);
        float n0 = h3_rng_normal(rng);
        float n1 = h3_rng_normal(rng);
        CHECK(std::isfinite(n0) && std::isfinite(n1));
        CHECK(rng.has_spare == 0);
        float fill[8];
        h3_rng_fill_normal(rng, fill, 8);
        float f2 = 0.f;
        for (float v : fill) {
            CHECK(std::isfinite(v));
            f2 += v * v;
        }
        CHECK(f2 > 0.f);
        H3Rng a, b;
        h3_rng_seed(a, 7);
        h3_rng_seed(b, 7);
        CHECK(h3_rng_u32(a) == h3_rng_u32(b));
    }
    {
        using namespace mvllm;
        H3SigmaSchedule sch;
        CHECK(!h3_schedule_build(0, sch));
        CHECK(h3_schedule_build(20, sch));
        CHECK(sch.steps == 20);
        CHECK(static_cast<int>(sch.video.size()) == 21);
        CHECK_NEAR(sch.video[0], 1.f, 1e-6);
        CHECK_NEAR(sch.audio[0], 1.f, 1e-6);
        CHECK_NEAR(sch.video[1], 0.995633185f, 1e-6);
        CHECK_NEAR(sch.audio[1], 0.982758582f, 1e-6);
        CHECK(sch.video[20] == 0.f && sch.audio[20] == 0.f);
        for (int i = 0; i < 20; ++i)
            CHECK(sch.video[static_cast<size_t>(i)] >= sch.video[static_cast<size_t>(i) + 1]);
        H3SigmaSchedule serve;
        CHECK(h3_serving_schedule_build(50, serve));
        CHECK(serve.steps == 50);
        CHECK_NEAR(serve.video[0], 1.f, 1e-5);
        CHECK_NEAR(serve.audio[0], 1.f, 1e-5);
        CHECK(serve.video[50] == 0.f && serve.audio[50] == 0.f);
        CHECK(serve.video[1] > serve.audio[1]);
        CHECK(!h3_serving_schedule_build(1, serve));
        CHECK_NEAR(h3_time_shift_sigma(0.5, 12.0, 12.0), 0.5, 1e-9);
        CHECK_NEAR(h3_time_shift_slope(0.5, 12.0, 12.0), 1.0, 1e-9);
        const double ts = h3_time_shift_sigma(0.5, 12.0, 3.0);
        CHECK(ts > 0.0 && ts < 0.5);

        H3DitSchedule dit;
        CHECK(dit.prepare(sch, true, false));
        CHECK(dit.steps() == 20);
        CHECK(dit.time_rows() == 40);
        CHECK(dit.video_row(0) == 0 && dit.audio_row(0) == 0);
        CHECK(dit.video_row(1) == 1 && dit.audio_row(1) == 2);
        CHECK(dit.visual_condition_row(0) == 39);
        CHECK(dit.audio_condition_row(0) == UINT32_MAX);
        CHECK(dit.video_row(-1) == UINT32_MAX);
        CHECK(static_cast<int>(dit.time_features().size()) == 40 * kH3TimeInput);
        CHECK_NEAR(dit.time_features()[0], 1.f, 1e-5);
        CHECK_NEAR(dit.time_features()[128], 0.f, 1e-5);

        H3Layout lay;
        CHECK(h3_layout_build(2, 2, 2, 2, 1, 5, lay));
        std::vector<uint32_t> rows(static_cast<size_t>(lay.seq_len), 99);
        uint8_t tags[2] = {1, 0};
        CHECK(dit.row_map(0, lay, tags, 2, rows.data(), lay.seq_len));
        CHECK(rows[0] == 0 * 3 + 1);
        CHECK(rows[1] == 0 * 3 + 0);
        bool saw_audio = false, saw_video = false;
        for (const auto &seg : lay.segments) {
            if (seg.kind == H3SegKind::Audio) {
                saw_audio = true;
                CHECK(rows[static_cast<size_t>(seg.start)] == 0 * 3 + 2);
            }
            if (seg.kind == H3SegKind::Video) {
                saw_video = true;
                CHECK(rows[static_cast<size_t>(seg.start)] == 0 * 3 + 0);
            }
        }
        CHECK(saw_audio && saw_video);
        CHECK(!dit.row_map(0, lay, tags, 1, rows.data(), lay.seq_len));

        float sig[3] = {1.f, 0.5f, 0.f};
        float sample[2] = {1.f, 2.f}, den[2] = {0.f, 0.f}, out[2] = {};
        CHECK(h3_res_step(out, sample, den, nullptr, 2, sig, 0, 2) == 1);
        CHECK_NEAR(out[0], 0.5f, 1e-5);
        CHECK_NEAR(out[1], 1.f, 1e-5);
        float oldd[2] = {0.25f, 0.5f};
        CHECK(h3_res_step(out, sample, den, oldd, 2, sig, 0, 2) == 0);
        CHECK(h3_res_step(out, sample, den, oldd, 2, sig, 1, 2) == 1);
        CHECK(std::isfinite(out[0]) && std::isfinite(out[1]));
    }
    {
        using namespace mvllm;
        const std::string dir = tmpdir();
        KvPersistConfig cfg;
        cfg.n_layers = 2;
        cfg.kv_lora = 4;
        cfg.qk_rope = 2;
        cfg.index_hd = 3;
        cfg.vocab = 16;
        cfg.has_index = {1, 0};
        KvPersist kp;
        std::string err;
        CHECK(kp.open(dir + "/cache.coli_kv", cfg, err) == Status::Ok);
        CHECK(kp.nrec() == 0);
        CHECK(kp.record_bytes() == 4 + 2 * (4 + 2) * 4 + 3 * 4);
        KvPersistRecord r0, r1;
        r0.L = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
        r0.R = {0.1f, 0.2f, 0.3f, 0.4f};
        r0.I = {9.f, 8.f, 7.f};
        r1.L.assign(8, 0.5f);
        r1.R.assign(4, -1.f);
        r1.I = {1.f, 0.f, -1.f};
        int hist[2] = {11, 22};
        KvPersistRecord recs[2] = {r0, r1};
        CHECK(kp.append(hist, 2, recs, 2, err) == Status::Ok);
        CHECK(kp.nrec() == 2);
        kp.close();

        KvPersist kp2;
        CHECK(kp2.open(dir + "/cache.coli_kv", cfg, err) == Status::Ok);
        std::vector<int> loaded;
        std::vector<KvPersistRecord> rows;
        CHECK(kp2.load(loaded, &rows, err) == 2);
        CHECK(loaded.size() == 2 && loaded[0] == 11 && loaded[1] == 22);
        CHECK(rows.size() == 2);
        CHECK_NEAR(rows[0].L[0], 1.f, 1e-6);
        CHECK_NEAR(rows[0].R[3], 0.4f, 1e-6);
        CHECK_NEAR(rows[0].I[2], 7.f, 1e-6);
        CHECK_NEAR(rows[1].L[7], 0.5f, 1e-6);
        CHECK(kp2.truncate(1, err) == Status::Ok);
        CHECK(kp2.nrec() == 1);
        loaded.clear();
        CHECK(kp2.load(loaded, nullptr, err) == 1);
        CHECK(loaded[0] == 11);
        CHECK(kp2.reset(err) == Status::Ok);
        CHECK(kp2.nrec() == 0);

        KvPersistConfig other = cfg;
        other.kv_lora = 8;
        KvPersist bad;
        CHECK(bad.open(dir + "/cache.coli_kv", other, err) == Status::Ok);
        CHECK(bad.nrec() == 0);
        std::vector<int> empty;
        CHECK(bad.load(empty, nullptr, err) == 0);

        KvPersistConfig noidx;
        noidx.n_layers = 1;
        noidx.kv_lora = 2;
        noidx.qk_rope = 1;
        noidx.vocab = 8;
        KvPersist kp3;
        CHECK(kp3.open(dir + "/plain.coli_kv", noidx, err) == Status::Ok);
        KvPersistRecord p;
        p.L = {1.f, 2.f};
        p.R = {3.f};
        int tok = 5;
        CHECK(kp3.append(&tok, 1, &p, 1, err) == Status::Ok);
        CHECK(kp3.append(&tok, 1, &p, 0, err) == Status::Ok);
        int two[2] = {5, 6};
        CHECK(kp3.append(two, 2, &p, 0, err) == Status::InvalidArgument);
    }
    {
        using namespace mvllm;
        const char payload[] = {'A', '\n', '\0', 'B'};
        std::string tool0 = mux_format_tool(7, nullptr, 0);
        CHECK(tool0 == "TOOL 7 0\n\n");
        std::string data = mux_format_data(7, payload, 4);
        CHECK(data.size() == std::string("DATA 7 4\n").size() + 4 + 1);
        CHECK(data.compare(0, 9, "DATA 7 4\n") == 0);
        CHECK(data[9] == 'A' && data[10] == '\n' && data[11] == '\0' && data[12] == 'B');
        CHECK(data[13] == '\n');
        CHECK(mux_hex_encode("ab", 2) == "6162");
        float lp[2] = {-0.1f, -0.2f};
        std::string tx[2] = {"a", "b"};
        CHECK(mux_format_topk(1, lp, tx, 2) == "TOPK 1 2 -0.1 61 -0.2 62\n");
        CHECK(mux_format_hwinfo(8, 16.0, 8.5, 0, 0.0, "cpu|x", "none") ==
              "HWINFO 8 16.00 8.50 0 0.00 cpu x|none\n");
        CHECK(mux_format_tiers(1, 2, 3, 1.5, 4.25) == "TIERS 1 2 3 1.50 4.25\n");
        CHECK(mux_emap_byte(0, 0) == 0);
        CHECK(mux_emap_byte(2, 0) == 0x80);
        CHECK(mux_emap_byte(1, 1) == ((1 << 6) | 1));
        CHECK(mux_emap_byte(1, 2) == ((1 << 6) | 2));
        uint8_t em[2] = {0, 0x80};
        CHECK(mux_format_emap(1, 2, em) == "EMAP 1 2 0080\n");
        uint8_t hit[3] = {1, 0, 1};
        CHECK(mux_format_hits(1, 3, hit) == "HITS 1 3 05\n");
        CHECK(mux_hits_hex(1, 3, hit) == "05");
        std::vector<uint8_t> bm;
        mux_hits_pack(1, 3, hit, bm);
        CHECK(bm.size() == 1 && bm[0] == 0x05);
        CHECK(mux_format_experts_json(0, 0, nullptr, nullptr, 0) ==
              "{\"rows\":0,\"cols\":0,\"map\":\"\",\"hits\":\"\",\"seq\":0}");
        CHECK(mux_format_experts_json(1, 2, em, hit, 4) ==
              "{\"rows\":1,\"cols\":2,\"map\":\"0080\",\"hits\":\"01\",\"seq\":4}");
        float ejent[2] = {1.5f, 2.f};
        CHECK(mux_format_experts_json(1, 2, em, hit, 4, ejent, 2) ==
              "{\"rows\":1,\"cols\":2,\"map\":\"0080\",\"hits\":\"01\",\"seq\":4,\"entropy\":[1.5,2]}");
        CHECK(experts_json(nullptr) ==
              mux_format_experts_json(0, 0, nullptr, nullptr, 0, nullptr, 0, serve_created()));
        CHECK(mux_format_experts_json(0, 0, nullptr, nullptr, 0, nullptr, 0, 9) ==
              "{\"rows\":0,\"cols\":0,\"map\":\"\",\"hits\":\"\",\"seq\":0,\"created\":9}");
        CHECK(mux_format_experts_json(0, 0, nullptr, nullptr, 0).find("created") ==
              std::string::npos);
        std::string perf = mux_format_perf(9, 0.5, 0.1, 0, 0, 0, 0, 0);
        CHECK(perf.find("PERF 9 ") == 0);
        CHECK(perf.back() == '\n');
        float ent[2] = {1.5f, 2.f};
        CHECK(mux_format_entropy(ent, 2) == "ENTROPY 1.5 2\n");
        CHECK(mux_format_gpus(0, nullptr, nullptr, nullptr) == "GPUS 0\n");
        double ug[1] = {1.5}, tg[1] = {8.0};
        int ex[1] = {3};
        CHECK(mux_format_gpus(1, ug, tg, ex) == "GPUS 1 1.50 8.00 3\n");
        CHECK(mux_format_repin(2, 7, 1, 0) == "REPIN 2 7 1 0\n");
        CHECK(mux_format_prof(0.5, 3, 2, 0.1, 0, 0.2, 0.05, 0.01, 2) ==
              "PROF 0.500 3 2 0.100 0.000 0.200 0.050 0.010 2\n");
        std::string turn = mux_format_turn_telem("HWINFO 1 1.00 1.00 0 0.00 cpu|none", 3, 0.1, 0,
                                                 0, 0, 0, 0, 0, ent, 2, 0, 0, 4, 0.0, 1.0, 1, 2,
                                                 em, 1, 3, hit);
        CHECK(turn.find("HWINFO 1") == 0);
        CHECK(turn.find("GPUS ") != std::string::npos);
        CHECK(turn.find("PERF 3 ") != std::string::npos);
        CHECK(turn.find("ENTROPY 1.5 2\n") != std::string::npos);
        CHECK(turn.find("TIERS 0 0 4 ") != std::string::npos);
        CHECK(turn.find("EMAP 1 2 0080\n") != std::string::npos);
        CHECK(turn.find("HITS 1 3 05\n") != std::string::npos);
        std::string hline = hwinfo_line();
        CHECK(hline.find("HWINFO ") == 0);
        CHECK(hline.back() == '\n');
    }
    {
        using namespace mvllm;
        std::string err;
        const uint8_t in4[12] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
        std::vector<uint8_t> id;
        CHECK(h3_resize_rgb24(in4, 1, 2, 2, 2, 2, id, err) == Status::Ok);
        CHECK(id.size() == 12);
        CHECK(id.data() != in4);
        for (int i = 0; i < 12; ++i)
            CHECK(id[static_cast<size_t>(i)] == in4[i]);
        const uint8_t ab[12] = {0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 255, 0};
        std::vector<uint8_t> up;
        CHECK(h3_resize_rgb24(ab, 1, 2, 2, 4, 4, up, err) == Status::Ok);
        CHECK(up.size() == 4 * 4 * 3);
        CHECK(up[0] == 0 && up[1] == 0 && up[2] == 0);
        CHECK(up[3 * 3] == 255 && up[3 * 3 + 1] == 0 && up[3 * 3 + 2] == 0);
        CHECK(up[3] == 64);
        CHECK(up[6] == 191);
        CHECK(h3_resize_rgb24(nullptr, 1, 2, 2, 2, 2, id, err) == Status::InvalidArgument);
        std::vector<float> fin(12), fout;
        for (int i = 0; i < 12; ++i)
            fin[static_cast<size_t>(i)] = in4[i] / 255.f;
        CHECK(h3_resize_rgb_f32(fin.data(), 1, 2, 2, 2, 2, fout, err) == Status::Ok);
        CHECK(fout.size() == 12);
        CHECK_NEAR(fout[0], fin[0], 1e-6);
        std::vector<float> fup;
        const float fab[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0};
        CHECK(h3_resize_rgb_f32(fab, 1, 2, 2, 4, 4, fup, err) == Status::Ok);
        CHECK_NEAR(fup[3], 0.25, 1e-5);
        CHECK_NEAR(fup[6], 0.75, 1e-5);
    }
    {
        using namespace mvllm;
        kv_fp8_lut_init();
        CHECK(kv_fp8_enc(0.f) == 0);
        CHECK(kv_fp8_enc(1.f) == 0x38);
        CHECK_NEAR(kv_fp8_lut(0x38), 1.f, 1e-6);
        CHECK(kv_fp8_enc(-448.f) == 0xfe);
        CHECK_NEAR(kv_fp8_lut(0xfe), -448.f, 1e-4);
        CHECK(kv_fp8_enc(std::numeric_limits<float>::quiet_NaN()) == 0);
        CHECK(kv_fp8_enc(1000.f) == 0x7e);
        CHECK_NEAR(kv_fp8_lut(0x7e), 448.f, 1e-4);
        CHECK(kv_fp8_lut(0x7f) == 0.f);
        float src[4] = {1.f, -2.f, 0.5f, 0.f};
        uint8_t q[4];
        float back[4];
        float sc = kv_fp8_quant_row(src, q, 4);
        CHECK(sc > 0.f);
        kv_fp8_dequant_row(q, sc, back, 4);
        for (int i = 0; i < 4; ++i)
            CHECK(std::fabs(back[i] - src[i]) < 0.05f);
        float z[2] = {0.f, 0.f};
        uint8_t zq[2] = {9, 9};
        CHECK_NEAR(kv_fp8_quant_row(z, zq, 2), 1.f, 1e-6);
        CHECK(zq[0] == 0 && zq[1] == 0);
        CHECK(kv_fp8_nscale(8, 0) == 1);
        CHECK(kv_fp8_nscale(8, 4) == 2);
        float scales[2];
        uint8_t qg[4];
        float bg[4];
        kv_fp8_quant_row_gs(src, qg, scales, 4, 2);
        kv_fp8_dequant_row_gs(qg, scales, bg, 4, 2);
        for (int i = 0; i < 4; ++i)
            CHECK(std::fabs(bg[i] - src[i]) < 0.05f);
    }
    {
        using namespace mvllm;
        MuxWireProfile prof;
        MuxCommand cmd;
        size_t used = 99;
        CHECK(mux_parse_command(nullptr, 0, prof, cmd, &used) == MuxRead::Eof);
        CHECK(used == 0);
        const char *stop = "STOP 7\n";
        CHECK(mux_parse_command(reinterpret_cast<const uint8_t *>(stop), 7, prof, cmd, &used) ==
              MuxRead::Ok);
        CHECK(used == 7 && cmd.kind == MuxCmd::Stop && cmd.id == 7);
        const char *part = "STOP 7";
        CHECK(mux_parse_command(reinterpret_cast<const uint8_t *>(part), 6, prof, cmd, &used) ==
              MuxRead::NeedMore);
        CHECK(used == 0);
        const char *ign = "PING 1\n";
        CHECK(mux_parse_command(reinterpret_cast<const uint8_t *>(ign), 7, prof, cmd, &used) ==
              MuxRead::Ignored);
        const char *img = "IMAGE 3 4 1 2\nABCD\n";
        CHECK(mux_parse_command(reinterpret_cast<const uint8_t *>(img), 19, prof, cmd, &used) ==
              MuxRead::Ok);
        CHECK(cmd.kind == MuxCmd::Image && cmd.id == 3 && cmd.grid_h == 1 && cmd.grid_w == 2);
        CHECK(cmd.payload.size() == 4 && cmd.payload[0] == 'A' && cmd.payload[3] == 'D');
        const char *sub = "SUBMIT 9 0 2 16 0.8 0.9\nhi\n";
        CHECK(mux_parse_command(reinterpret_cast<const uint8_t *>(sub), 27, prof, cmd, &used) ==
              MuxRead::Ok);
        CHECK(cmd.kind == MuxCmd::Submit && cmd.id == 9 && cmd.slot == 0);
        CHECK(cmd.max_tokens == 16);
        CHECK_NEAR(cmd.temperature, 0.8f, 1e-6);
        CHECK_NEAR(cmd.top_p, 0.9f, 1e-6);
        CHECK(cmd.payload.size() == 2 && cmd.payload[0] == 'h' && cmd.payload[1] == 'i');
        const char *bad = "STOP 7 extra\n";
        CHECK(mux_parse_command(reinterpret_cast<const uint8_t *>(bad), 13, prof, cmd, &used) ==
              MuxRead::BadRequest);
    }
    {
        using namespace mvllm;
        CHECK(route_usage_hash("kimi_k3") == 1820578806u);
        RouteUsage ru;
        std::string err;
        CHECK(ru.init("kimi_k3", 5, 8, err) == Status::Ok);
        CHECK(ru.n_layers() == 5 && ru.n_experts() == 8);
        ru.drop_row(0);
        int ids[3] = {2, 5, 2};
        ru.count(3, ids, 3);
        CHECK(ru.get(3, 2) == 2 && ru.get(3, 5) == 1);
        CHECK(ru.get(0, 1) == 0);
        const std::string dir = tmpdir();
        const std::string path = dir + "/.coli_usage";
        CHECK(ru.save(path, err));
        std::ifstream in(path);
        std::string line;
        std::getline(in, line);
        CHECK(line == "-1 5 8");
        std::getline(in, line);
        CHECK(line == "-2 1 1820578806");
        RouteUsage ru2;
        CHECK(ru2.init("kimi_k3", 5, 8, err) == Status::Ok);
        CHECK(ru2.load(path, false, err) == 2);
        CHECK(ru2.get(3, 2) == 2 && ru2.get(3, 5) == 1);
        RouteUsage other;
        CHECK(other.init("glm_moe_dsa", 5, 8, err) == Status::Ok);
        CHECK(other.load(path, false, err) == -1);
        CHECK(other.load(path, true, err) == 2);
        RouteUsage empty;
        CHECK(empty.init("kimi_k3", 2, 4, err) == Status::Ok);
        CHECK(empty.save(dir + "/empty.coli_usage", err));
        std::ifstream ez(dir + "/empty.coli_usage", std::ios::binary | std::ios::ate);
        CHECK(ez.tellg() == 0);
        ru.decay(0.5);
        CHECK(ru.get(3, 2) == 1 && ru.get(3, 5) == 1);
    }
    {
        using namespace mvllm;
        CHECK(h3_adaln_out(1) == 18);
        CHECK(h3_adaln_out(0) == 0);
        float x[2] = {1.f, 0.f};
        h3_silu(x, 2);
        CHECK_NEAR(x[0], 1.f / (1.f + std::exp(-1.f)), 1e-6);
        CHECK_NEAR(x[1], 0.f, 1e-6);
        const float feat[2] = {1.f, 0.5f};
        const float win[4] = {1.f, 0.f, 0.f, 1.f};
        const float bin[2] = {0.1f, -0.2f};
        const float wout[4] = {0.5f, 1.f, -1.f, 0.25f};
        const float bout[2] = {0.f, 0.3f};
        const float wad[4] = {1.f, 0.5f, 0.f, 2.f};
        const float bad[2] = {0.f, -0.1f};
        float temb[2] = {}, mod[2] = {};
        CHECK(h3_time_embed(feat, 1, 2, win, bin, 2, wout, bout, 2, temb));
        CHECK_NEAR(temb[0], 0.375678211f, 1e-5);
        CHECK_NEAR(temb[1], -0.184072331f, 1e-5);
        CHECK(h3_adaln_mod(temb, 1, 2, wad, bad, 2, mod));
        CHECK_NEAR(mod[0], 0.283642054f, 1e-5);
        CHECK_NEAR(mod[1], -0.468144655f, 1e-5);
        CHECK(!h3_time_embed(feat, 0, 2, win, bin, 2, wout, bout, 2, temb));
        const float ident[4] = {1.f, 0.f, 0.f, 1.f};
        const float fx[2] = {1.f, 0.f};
        float t2[2] = {};
        CHECK(h3_time_embed(fx, 1, 2, ident, nullptr, 2, ident, nullptr, 2, t2));
        const float s1 = 1.f / (1.f + std::exp(-1.f));
        const float s2 = s1 / (1.f + std::exp(-s1));
        CHECK_NEAR(t2[0], s2, 1e-5);
        CHECK_NEAR(t2[1], 0.f, 1e-6);
    }
    {
        using namespace mvllm;
        uint8_t sel[50];
        CHECK(h3_dit_reuse_schedule(20, 3, sel, 50) == 8);
        const int agg[] = {0, 3, 6, 9, 12, 15, 18, 19};
        for (int s = 0; s < 20; ++s) {
            int exp = 0;
            for (int a : agg)
                exp |= (s == a);
            CHECK(sel[s] == exp);
        }
        CHECK(h3_dit_reuse_schedule(20, 2, sel, 50) == 11);
        for (int s = 0; s < 20; ++s)
            CHECK(sel[s] == ((s % 2 == 0) || s == 19));
        CHECK(h3_dit_reuse_schedule(50, 3, sel, 50) == 18);
        CHECK(h3_dit_reuse_schedule(20, 1, sel, 50) == 20);
        CHECK(h3_dit_reuse_schedule(20, 3, sel, 19) == -1);
        CHECK(h3_parse_reuse_steps(20, nullptr, sel) == 0);
        CHECK(h3_parse_reuse_steps(20, "0,3,6,19", sel) == 4);
        CHECK(sel[0] && sel[3] && sel[6] && sel[19] && !sel[1]);
        CHECK(h3_parse_reuse_steps(20, "0,3,6", sel) == -1);
        CHECK(h3_parse_reuse_steps(20, "0,3,3,19", sel) == -1);
    }
    {
        using namespace mvllm;
        HwInfo hi = hw_probe();
        CHECK(hi.cores > 0);
        CHECK(hi.ram_total_gb > 0.0);
        CHECK(hi.ngpu >= 0);
        CHECK(hi.vram_total_gb >= 0.0);
        if (hi.ngpu > 0)
            CHECK(!hi.gpu.empty());
        CHECK(!hi.cpu.empty());
        CHECK(rss_gb() >= 0.0);
        std::string line = mux_format_hwinfo(hi.cores, hi.ram_total_gb, hi.ram_avail_gb, hi.ngpu,
                                             hi.vram_total_gb, hi.cpu, hi.gpu.empty() ? "none" : hi.gpu);
        CHECK(line.find("HWINFO ") == 0);
        CHECK(line.back() == '\n');
    }
    {
        using namespace mvllm;
        CHECK(kv_tq_row_bytes(512, 4) == 224);
        CHECK(kv_tq_row_bytes(64, 4) == 28);
        CHECK(kv_q4_row_bytes(512) == 256);
        float a[4] = {1.f, 2.f, 3.f, 4.f};
        kv_tq_fwht(a, 4);
        kv_tq_fwht(a, 4);
        CHECK_NEAR(a[0], 1.f, 1e-5);
        CHECK_NEAR(a[3], 4.f, 1e-5);
        float src[4] = {0.7f, -1.2f, 0.3f, 2.1f};
        uint8_t packed[16] = {};
        float back[4] = {};
        float rad = kv_tq_quant_row(src, packed, 4, 4);
        CHECK_NEAR(rad, 2.53574443f, 1e-5);
        kv_tq_dequant_row(packed, rad, back, 4, 4);
        float num = 0.f, den = 0.f;
        for (int i = 0; i < 4; ++i) {
            float d = back[i] - src[i];
            num += d * d;
            den += src[i] * src[i];
        }
        CHECK(std::sqrt(num / den) < 0.2f);
        float z[4] = {};
        uint8_t zp[16];
        CHECK(kv_tq_quant_row(z, zp, 4, 4) == 0.f);
        float zb[4] = {9, 9, 9, 9};
        kv_tq_dequant_row(zp, 0.f, zb, 4, 4);
        CHECK(zb[0] == 0.f && zb[3] == 0.f);
        uint8_t q4[2] = {};
        float qrad = kv_q4_quant_row(src, q4, 4);
        CHECK(qrad > 0.f);
        float qb[4] = {};
        kv_q4_dequant_row(q4, qrad, qb, 4);
        float qn = 0.f;
        for (int i = 0; i < 4; ++i) {
            float d = qb[i] - src[i];
            qn += d * d;
        }
        CHECK(std::sqrt(qn / den) < 0.25f);
        CHECK(kv_tq_quant_row(src, packed, 3, 4) == 0.f);
    }
    {
        using namespace mvllm;
        H3TokenReduce cfg;
        std::string err;
        CHECK(h3_token_reduce_configure(cfg, false, 2, 2, 4, 3, 4, 7, nullptr, nullptr, nullptr,
                                        err));
        CHECK(!cfg.enabled);
        CHECK(h3_token_reduce_configure(cfg, true, 2, 2, 4, 3, 4, 7, nullptr, nullptr, nullptr,
                                        err));
        CHECK(cfg.enabled);
        CHECK(cfg.begin == 4 && cfg.end == 30);
        CHECK(cfg.early_steps == 10 && cfg.early_end == 40);
        CHECK(cfg.spatial_h == 1 && cfg.spatial_w == 2 && cfg.reduced_w == 1);
        CHECK(cfg.reduced_video_rows == 2 && cfg.reduced_sequence == 5 && cfg.baseline_rows == 2);
        uint32_t a = 0, b = 0;
        h3_token_pool_sources(cfg, 2, &a, &b);
        CHECK(a == 2 && b == 2);
        h3_token_pool_sources(cfg, 3, &a, &b);
        CHECK(a == 3 && b == 4);
        h3_token_pool_sources(cfg, 4, &a, &b);
        CHECK(a == 5 && b == 6);
        CHECK(h3_token_reduced_parent(cfg, 3) == 3);
        CHECK(h3_token_reduced_parent(cfg, 4) == 3);
        CHECK(h3_token_reduced_parent(cfg, 5) == 4);
        float x[2] = {1.f, 3.f}, y[2] = {3.f, 5.f}, m[2];
        h3_token_pool_mean(x, y, m, 2);
        CHECK_NEAR(m[0], 2.f, 1e-6);
        CHECK_NEAR(m[1], 4.f, 1e-6);
        CHECK(h3_token_reduce_configure(cfg, true, 2, 2, 4, 3, 4, 8, nullptr, nullptr, nullptr,
                                        err) == false);
        CHECK(h3_token_reduce_configure(cfg, true, 2, 2, 4, 3, 4, 7, "4:30", "0", "1.25", err));
        CHECK(cfg.early_steps == 0 && cfg.scale == 1.25f);
    }
    {
        using namespace mvllm;
        KvPersistConfig cfg;
        cfg.n_layers = 2;
        cfg.kv_lora = 4;
        cfg.qk_rope = 2;
        cfg.vocab = 16;
        KvPersistV2 kp;
        std::string err;
        const std::string dir = tmpdir();
        CHECK(kp.open(dir + "/cache.coli_kv2", cfg, err) == Status::Ok);
        CHECK(kp.nrec() == 0);
        CHECK(kp.record_bytes() == 4 + 2 * (4 + 4 + 2 + 4));
        KvPersistRecord r;
        r.L = {1.f, -2.f, 0.5f, 0.f, 0.25f, 0.25f, 0.25f, 0.25f};
        r.R = {0.1f, 0.2f, -0.3f, 0.4f};
        int hist = 11;
        CHECK(kp.append(&hist, 1, &r, 1, err) == Status::Ok);
        CHECK(kp.nrec() == 1);
        kp.close();
        KvPersistV2 kp2;
        CHECK(kp2.open(dir + "/cache.coli_kv2", cfg, err) == Status::Ok);
        std::vector<int> loaded;
        std::vector<KvPersistRecord> rows;
        CHECK(kp2.load(loaded, &rows, err) == 1);
        CHECK(loaded[0] == 11);
        CHECK(rows[0].L.size() == 8 && rows[0].R.size() == 4);
        CHECK(std::fabs(rows[0].L[0] - 1.f) < 0.08f);
        KvPersistConfig other = cfg;
        other.kv_lora = 8;
        KvPersistV2 bad;
        CHECK(bad.open(dir + "/cache.coli_kv2", other, err) == Status::Ok);
        CHECK(bad.nrec() == 0);
    }
    {
        using namespace mvllm;
        CHECK(moe_router_pick(3, 0, 8, 1, nullptr) == 3);
        std::string warn;
        CHECK(moe_router_pick(-1, 2, 8, 4, &warn) == 2);
        CHECK(!warn.empty());
        CHECK(moe_router_pick(-1, 9, 8, 4, &warn) == 0);
        float sc[4] = {0.1f, 0.8f, 0.3f, 0.5f};
        CHECK(moe_argmax(sc, 4) == 1);
        int idx[2];
        float w[2];
        CHECK(moe_topk_pick(sc, 4, 2, idx, w) == 2);
        CHECK(idx[0] == 1 && idx[1] == 3);
        CHECK_NEAR(w[0] + w[1], 1.f, 1e-5);
        CHECK(w[0] > w[1]);
        float nan[4] = {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN()};
        CHECK(moe_argmax(nan, 4) == -1);
        CHECK(moe_topk_pick(nan, 4, 2, idx, w) == 2);
        CHECK(idx[0] == 0 && idx[1] == 1);
        CHECK_NEAR(w[0], 0.5f, 1e-6);
        CHECK_NEAR(w[1], 0.5f, 1e-6);
    }
    {
        using namespace mvllm;
        const int ids[2] = {2, 5};
        const float gates[2] = {0.6f, 0.4f};
        CHECK(route_trace_format(0, 0, 3, ids, gates, 2) == "0 0 3 2:0.6000 5:0.4000\n");
        RouteTraceEvent ev;
        CHECK(route_trace_parse("0 0 3 2:0.6000 5:0.4000", ev));
        CHECK(ev.call == 0 && ev.row == 0 && ev.layer == 3);
        CHECK(ev.ids.size() == 2 && ev.ids[0] == 2 && ev.ids[1] == 5);
        CHECK_NEAR(ev.gates[0], 0.6f, 1e-4);
        CHECK(!route_trace_parse("junk", ev));
        std::string sink, err;
        RouteTrace tr;
        CHECK(tr.attach(&sink, err));
        CHECK(tr.emit(0, 3, ids, gates, 2, err));
        tr.end();
        CHECK(tr.call() == 1);
        CHECK(sink == "0 0 3 2:0.6000 5:0.4000\n");
        const std::string dir = tmpdir();
        RouteTrace tf;
        CHECK(tf.open(dir + "/route.txt", err));
        CHECK(tf.emit(1, 2, ids, gates, 2, err));
        tf.close();
        std::ifstream in(dir + "/route.txt");
        std::string line;
        std::getline(in, line);
        CHECK(line == "0 1 2 2:0.6000 5:0.4000");
    }
    {
        using namespace mvllm;
        KvPersistConfig cfg;
        cfg.n_layers = 1;
        cfg.kv_lora = 4;
        cfg.qk_rope = 4;
        cfg.vocab = 16;
        KvPersistV3Options opt;
        KvPersistV3 kp;
        std::string err;
        const std::string dir = tmpdir();
        CHECK(kp.open(dir + "/cache.coli_kv3", cfg, opt, err) == Status::Ok);
        CHECK(kp.nrec() == 0);
        CHECK(kp.codec() == 0 && kp.bits() == 4);
        CHECK(kp.record_bytes() == 16);
        KvPersistRecord r;
        r.L = {0.7f, -1.2f, 0.3f, 2.1f};
        r.R = {0.5f, 0.5f, -0.5f, 0.25f};
        int hist = 9;
        CHECK(kp.append(&hist, 1, &r, 1, err) == Status::Ok);
        CHECK(kp.nrec() == 1);
        kp.close();
        KvPersistV3 kp2;
        CHECK(kp2.open(dir + "/cache.coli_kv3", cfg, opt, err) == Status::Ok);
        std::vector<int> loaded;
        std::vector<KvPersistRecord> rows;
        CHECK(kp2.load(loaded, &rows, err) == 1);
        CHECK(loaded[0] == 9);
        CHECK(rows[0].L.size() == 4);
        float den = 0.f, num = 0.f;
        for (int i = 0; i < 4; ++i) {
            float d = rows[0].L[static_cast<size_t>(i)] - r.L[static_cast<size_t>(i)];
            num += d * d;
            den += r.L[static_cast<size_t>(i)] * r.L[static_cast<size_t>(i)];
        }
        CHECK(std::sqrt(num / den) < 0.25f);
        KvPersistV3Options q4;
        q4.codec = 1;
        KvPersistV3 mismatch;
        CHECK(mismatch.open(dir + "/cache.coli_kv3", cfg, q4, err) == Status::Ok);
        CHECK(mismatch.nrec() == 0);
    }
    {
        using namespace mvllm;
        const float x[2] = {1.f, 0.f};
        const float W[4] = {1.f, 0.f, 0.f, 1.f};
        int8_t wq[4];
        float wsc[2];
        int8_quant_rows(W, 2, 2, wq, wsc);
        float y[2] = {};
        int8_dyn_gemm(y, x, wq, wsc, nullptr, 1, 2, 2);
        CHECK_NEAR(y[0], 1.f, 1e-5);
        CHECK_NEAR(y[1], 0.f, 1e-5);
        int8_t q[2];
        float sc = 0.f;
        int8_quant_row(x, 2, q, &sc);
        CHECK(q[0] == 127 && q[1] == 0);
        CHECK_NEAR(sc, 1.f / 127.f, 1e-6);
        const float bias[2] = {0.5f, -0.25f};
        int8_dyn_gemm(y, x, wq, wsc, bias, 1, 2, 2);
        CHECK_NEAR(y[0], 1.5f, 1e-5);
        CHECK_NEAR(y[1], -0.25f, 1e-5);
        float z[2] = {};
        int8_t zq[2];
        float zs = 9.f;
        int8_quant_row(z, 2, zq, &zs);
        CHECK(zq[0] == 0 && zq[1] == 0 && zs == 0.f);
    }
    {
        using namespace mvllm;
        MuxSubmit s;
        CHECK(mux_submit_parse("SUBMIT 42 3 17 64 0.7 0.95", s));
        CHECK(s.id == 42 && s.slot == 3 && s.bytes == 17 && s.max_tokens == 64);
        CHECK_NEAR(s.temperature, 0.7f, 1e-6);
        CHECK_NEAR(s.top_p, 0.95f, 1e-6);
        CHECK(s.gbytes == 0 && s.logprobs == 0 && s.tok_ids == 0);
        CHECK(mux_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512 logprobs=32 ids=1", s));
        CHECK(s.gbytes == 512 && s.logprobs == 32 && s.tok_ids == 1);
        CHECK(!mux_submit_parse("SUBMIT 42 3 17 64 0.7 0.95 512 foo=1", s));
        MuxSubmitExt ext;
        CHECK(mux_submit_ext("", ext) == 0);
        CHECK(mux_submit_ext("logprobs=5", ext) == 1 && ext.logprobs == 5);
        CHECK(mux_submit_ext("logprobs=33", ext) == -1);
        int ids[8];
        CHECK(mux_ids_parse("  5\n7\t9  ", 9, ids, 8, 100) == 3);
        CHECK(ids[0] == 5 && ids[1] == 7 && ids[2] == 9);
        CHECK(mux_ids_parse("1 2 3", 5, ids, 2, 100) == 2);
        CHECK(mux_ids_parse("99", 2, ids, 8, 50) == -1);
    }
    {
        using namespace mvllm;
        CHECK_NEAR(e8m0_decode(0x7e), 0.5f, 1e-6);
        CHECK_NEAR(e8m0_decode(0x7f), 1.f, 1e-6);
        CHECK_NEAR(e8m0_decode(0x80), 2.f, 1e-6);
        CHECK(std::isnan(e8m0_decode(0xff)));
        const float *tab = e8m0_table();
        CHECK(tab && tab[0x7f] == e8m0_decode(0x7f));
        CHECK(e4m3fn_encode(1.f) != 0);
        CHECK_NEAR(e4m3fn_decode(e4m3fn_encode(1.f)), 1.f, 1e-6);
        CHECK(e4m3fn_encode(std::numeric_limits<float>::quiet_NaN()) == 0x7f);
        std::vector<float> in(128, 1.f), out(128, 0.f);
        uint8_t sc = 0;
        CHECK(fp8_activation_qdq(out.data(), &sc, in.data(), 128, 128) == 0);
        CHECK(sc == 119);
        for (float v : out)
            CHECK_NEAR(v, 1.f, 1e-5);
        std::vector<float> in4(32, 1.f), out4(32, 0.f);
        uint8_t sc4 = 0;
        CHECK(fp4_activation_qdq(out4.data(), &sc4, in4.data(), 32, 32) == 0);
        CHECK(sc4 == 125);
        float pin[20], pout[20];
        uint8_t psc = 0;
        for (int i = 0; i < 20; ++i)
            pin[i] = 1.f;
        CHECK(fp8_activation_qdq(pout, &psc, pin, 20, 64) == 0);
        CHECK(psc == 119);
        CHECK(fp8_activation_qdq(pout, &psc, pin, 0, 64) == -1);
        std::vector<float> x(128, 1.f), y(2, 0.f), ya(2, 0.f), yb(2, 0.f);
        std::vector<uint8_t> w(2 * 128, e4m3fn_encode(1.f));
        uint8_t tsc[2] = {0x7f, 0x7f};
        CHECK(fp8_matvec(y.data(), w.data(), tsc, 2, 128, x.data()) == 0);
        CHECK(std::isfinite(y[0]) && std::isfinite(y[1]));
        CHECK(y[0] != 0.f);
        CHECK(fp8_dual_matvec(ya.data(), yb.data(), w.data(), tsc, w.data(), tsc, 2, 128,
                              x.data()) == 0);
        CHECK_NEAR(ya[0], y[0], 1e-5);
        CHECK_NEAR(yb[1], y[1], 1e-5);
        CHECK(fp8_matvec(y.data(), w.data(), tsc, 2, 127, x.data()) == -1);
        std::vector<float> xhat(128, 1.f), ypre(2, 0.f);
        CHECK(fp8_matvec_pre(ypre.data(), w.data(), tsc, 2, 128, xhat.data()) == 0);
        CHECK(std::isfinite(ypre[0]));
        std::vector<float> xb(2 * 128, 1.f), ybch(2 * 2, 0.f);
        CHECK(fp8_matmul_batch(ybch.data(), w.data(), tsc, 2, 128, xb.data(), 2) == 0);
        CHECK_NEAR(ybch[0], y[0], 1e-4);
        CHECK(fp8_matmul_batch_pre(ybch.data(), w.data(), tsc, 2, 128, xb.data(), 2) == 0);
        CHECK(fp8_matmul_batch(ybch.data(), w.data(), tsc, 2, 128, xb.data(), 0) == -1);
        unsetenv("COLI_LOGIT_DUMP");
        unsetenv("MVLLM_LOGIT_DUMP");
        unsetenv("COLI_LOGIT_GAP");
        unsetenv("MVLLM_LOGIT_GAP");
        CHECK(logit_dump_maybe(y.data(), 2) == 0);
        CHECK(logit_gap_maybe(0, y.data(), 2) == 0);
        std::vector<uint8_t> w4(2 * 64, 0x22);
        uint8_t s4[2 * 4];
        for (int i = 0; i < 8; ++i)
            s4[i] = 0x7f;
        std::vector<float> y4(2, 0.f), ya4(2, 0.f), yb4(2, 0.f);
        CHECK(fp4_matvec(y4.data(), w4.data(), s4, 2, 128, x.data()) == 0);
        CHECK(std::isfinite(y4[0]) && y4[0] != 0.f);
        CHECK(fp4_dual_matvec(ya4.data(), yb4.data(), w4.data(), s4, w4.data(), s4, 2, 128,
                              x.data()) == 0);
        CHECK_NEAR(ya4[0], y4[0], 1e-5);
        CHECK(fp4_matvec(y4.data(), w4.data(), s4, 2, 64, x.data()) == -1);
        std::vector<uint8_t> w16(16 * 64, 0x22);
        std::vector<uint8_t> s16(16 * 4, 0x7f);
        std::vector<float> xq(128, 1.f), yr(16, 0.f);
        CHECK(fp4_matvec_rows16(yr.data(), w16.data(), s16.data(), 16, 128, xq.data()) == 0);
        CHECK(std::isfinite(yr[0]) && yr[0] != 0.f);
        CHECK(fp4_matvec_rows16(yr.data(), w16.data(), s16.data(), 15, 128, xq.data()) == -1);
        const int hid = 2;
        std::vector<float> mod(static_cast<size_t>(1 * 3 * 6 * hid), 0.f);
        for (int m = 0; m < 3; ++m) {
            for (int sl : {2, 5}) {
                const size_t b = static_cast<size_t>((m * 6 + sl) * hid);
                mod[b] = 1.f;
                mod[b + 1] = 1.f;
            }
        }
        CHECK_NEAR(h3_adaln_gate_score(mod.data(), 1, hid), 1.0, 1e-9);
        CHECK(h3_adaln_gate_score(nullptr, 1, hid) < 0);
        uint8_t act[3] = {1, 1, 1};
        const double scs[3] = {0.5, 0.1, -1.0};
        CHECK(h3_dit_prune_blocks(act, scs, 3, 0.2) == 1);
        CHECK(act[0] == 1 && act[1] == 0 && act[2] == 0);
    }
    {
        using namespace mvllm;
        CHECK(bf16_round(1.00390625f) == 1.0f);
        CHECK(bf16_round(1.01171875f) == 1.015625f);
        CHECK(bf16_decode(bf16_encode(1.0f)) == 1.0f);
        float h[4] = {1.f, 2.f, 3.f, 4.f};
        CHECK(hadamard_bf16(h, 4) == 0);
        CHECK_NEAR(h[0], 5.f, 1e-5);
        CHECK_NEAR(h[1], -1.f, 1e-5);
        CHECK_NEAR(h[2], -2.f, 1e-5);
        CHECK_NEAR(h[3], 0.f, 1e-5);
        CHECK(hadamard_bf16(h, 3) == -1);
    }
    {
        using namespace mvllm;
        const float logits[] = {0.f, 3.f, 1.f, 2.f};
        uint64_t rng = 0;
        CHECK(nuc_pick(logits, 4, 0.f, 1.f, -1, &rng) == 1);
        float p[4];
        CHECK(nuc_dist_build(logits, 4, 1.f, 0.5f, p) == 1);
        CHECK_NEAR(p[1], 1.f, 1e-5);
        CHECK(p[0] == 0.f && p[2] == 0.f && p[3] == 0.f);
        CHECK(nuc_dist_sample(p, 4, -1, &rng) == 1);
        float nanlo[] = {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::quiet_NaN()};
        CHECK(nuc_dist_build(nanlo, 2, 1.f, 1.f, p) == 1);
        CHECK_NEAR(p[0], 1.f, 1e-6);
        CHECK(p[1] == 0.f);
        double u = nuc_rng_u01(&rng);
        CHECK(u >= 0.0 && u < 1.0);
    }
    {
        using namespace mvllm;
        int stops[] = {2, 5};
        uint8_t specials[8] = {};
        specials[7] = 1;
        StopSet s;
        s.arm(stops, 2, 1, specials, 8, false);
        CHECK(s.size() == 4);
        CHECK(s.data()[0] == 2 && s.data()[1] == 5 && s.data()[2] == 1 && s.data()[3] == 7);
        CHECK(s.contains(7) && !s.contains(3));
        CHECK(s.specials_added() == 1);
        s.arm(stops, 2, 1, specials, 8, true);
        CHECK(s.size() == 1 && s.data()[0] == 1 && s.specials_added() == 0);
        s.arm(nullptr, 0, -1, nullptr, 0, false);
        CHECK(s.size() == 0);
        CHECK(s.empty());
        s.arm(stops, 2, 1, specials, 8, false);
        CHECK(!s.empty());
        s.arm(stops, 2, -1, specials, 8, true);
        CHECK(s.empty());
        s.arm(stops, 2, 1, specials, 8, false);
        CHECK(!s.empty());
        s.clear();
        CHECK(s.empty() && s.size() == 0 && s.specials_added() == 0);
        CHECK(!s.contains(2) && !s.contains(1));
    }
    {
        using namespace mvllm;
        Gbnf g;
        std::string e;
        CHECK(g.compile("root ::= \"yes\"", e) == Status::Ok);
        char buf[8] = {};
        CHECK(gbnf_forced_bytes(g, buf, 8) == 3);
        CHECK(std::string(buf, 3) == "yes");
        Gbnf g2;
        CHECK(g2.compile("root ::= \"yes\"", e) == Status::Ok);
        auto dec = [](int id) -> std::string {
            if (id == 1)
                return "y";
            if (id == 2)
                return "e";
            if (id == 3)
                return "s";
            if (id == 4)
                return "n";
            return "";
        };
        int toks[8];
        CHECK(gbnf_forced_tokens(g2, dec, 5, toks, 8) == 3);
        CHECK(toks[0] == 1 && toks[1] == 2 && toks[2] == 3);
        Gbnf dead;
        CHECK(gbnf_forced_bytes(dead, buf, 8) == 0);
    }
    {
        using namespace mvllm;
        KvPrefix p;
        CHECK(p.empty());
        CHECK(!p.full());
        CHECK(p.alloc(16));
        CHECK(!p.full());
        const int turn1[] = {5, 6, 7};
        const int turn2[] = {5, 6, 7, 8, 9};
        const int other[] = {5, 6, 99, 8, 9};
        CHECK(p.reuse(turn2, 5) == 0);
        p.record(turn1, 0, 3);
        CHECK(p.len() == 3);
        CHECK(!p.empty());
        CHECK(p.reuse(turn2, 5) == 3);
        CHECK(p.reuse(other, 5) == 0);
        CHECK(p.reuse(turn1, 3) == 0);
        const int shorter[] = {5, 6};
        CHECK(p.reuse(shorter, 2) == 0);
        const int gen[] = {8, 9};
        p.record(gen, 3, 2);
        CHECK(p.len() == 5);
        const int turn3[] = {5, 6, 7, 8, 9, 10};
        CHECK(p.reuse(turn3, 6) == 5);
        p.taint();
        CHECK(p.reuse(turn3, 6) == 0);
        p.clear();
        CHECK(p.len() == 0 && !p.tainted());
        CHECK(p.empty());
        CHECK(!p.full());
        CHECK(p.alloc(4));
        p.record(turn1, 0, 3);
        CHECK(!p.full());
        const int spill[] = {1, 2, 3};
        p.record(spill, 3, 3);
        CHECK(p.len() == 0);
        CHECK(!p.full());
        CHECK(p.alloc(3));
        p.record(turn1, 0, 3);
        CHECK(p.full());
    }
    {
        using namespace mvllm;
        const float lo[] = {0.f, 1.f, 0.f};
        int am = 0;
        const double got = logprob_target(lo, 3, 1, &am);
        const double want = -std::log(1.0 + 2.0 / std::exp(1.0));
        CHECK(am == 1);
        CHECK(std::fabs(got - want) < 1e-12);
        CHECK(logprob_target(lo, 3, 0, &am) < got);
        CHECK(am == 0);
        CHECK(logprob_argmax(lo, 3) == 1);
        const float tie[] = {1.f, 1.f, 0.f};
        CHECK(logprob_argmax(tie, 3) == 0);
        CHECK(logprob_argmax(nullptr, 3) == -1);
        CHECK(logprob_argmax(lo, 0) == -1);
        CHECK(model_type_is_glm("GlmForCausalLM"));
        CHECK(model_type_is_glm("foo_GLM_bar"));
        CHECK(!model_type_is_glm("llama"));
        CHECK(!model_type_is_glm(nullptr));
    }
    {
        using namespace mvllm;
        CHECK(std::string(h3_segment_name(H3SegKind::Text)) == "text");
        CHECK(std::string(h3_segment_name(H3SegKind::Cond)) == "cond");
        CHECK(std::string(h3_segment_name(H3SegKind::RefImage)) == "ref_img");
        CHECK(std::string(h3_segment_name(H3SegKind::RefAudio)) == "ref_audio");
        CHECK(std::string(h3_segment_name(H3SegKind::Audio)) == "audio");
        CHECK(std::string(h3_segment_name(H3SegKind::Video)) == "video");
        H3SegKind k = H3SegKind::Text;
        CHECK(h3_seg_kind_from_name("ref_img", &k) && k == H3SegKind::RefImage);
        CHECK(!h3_seg_kind_from_name("unknown", &k));
        int sig[kH3SigN], sig2[kH3SigN];
        h3_layout_sig_pack(12, 2, 2, 2, 8, sig);
        CHECK(sig[0] == 12 && sig[4] == 8);
        h3_layout_sig_pack(12, 2, 2, 2, 8, sig2);
        CHECK(h3_layout_sig_equal(sig, sig2));
        sig2[1] = 9;
        CHECK(!h3_layout_sig_equal(sig, sig2));
    }
    {
        using namespace mvllm;
        float sequence_a[4 * 3] = {};
        float sequence_b[4 * 3] = {};
        float *a2 = kv_row(sequence_a, 2, 3);
        float *b1 = kv_row(sequence_b, 1, 3);
        CHECK(a2 == &sequence_a[6]);
        CHECK(b1 == &sequence_b[3]);
        a2[0] = 20.f;
        b1[2] = 12.f;
        CHECK(sequence_a[6] == 20.f && sequence_b[5] == 12.f);
        CHECK(sequence_a[5] == 0.f && sequence_b[6] == 0.f);
        float storage[5 * 7] = {};
        CHECK(kv_row(static_cast<const float *>(storage), 4, 7) == &storage[28]);
        CHECK(kv_row_bytes(2, 3, 4) == 24);
        CHECK(kv_row_ok(2, 3, 12) && !kv_row_ok(2, 3, 8));
        CHECK(kv_row(static_cast<float *>(nullptr), 0, 3) == nullptr);
        uint8_t bytes[8] = {};
        CHECK(kv_row8(bytes, 1, 4) == &bytes[4]);
    }
    {
        using namespace mvllm;
        uint32_t cp = 0;
        const unsigned char A[] = {'A'};
        CHECK(utf8_next(A, 1, 0, &cp) == 1 && cp == 0x41);
        const unsigned char e[] = {0xC3, 0xA9};
        CHECK(utf8_next(e, 2, 0, &cp) == 2 && cp == 0xE9);
        std::string grin = utf8_encode_cp(0x1F600);
        CHECK(grin.size() == 4);
        CHECK(static_cast<unsigned char>(grin[0]) == 0xF0);
        char buf[8] = {};
        CHECK(utf8_put(buf, 0x41) == 1 && buf[0] == 'A');
        uint32_t cps[8];
        CHECK(utf8_decode_all(e, 2, cps, 8) == 1 && cps[0] == 0xE9);
        CHECK(utf8_next(A, 1, 5, &cp) == 0);
        CHECK(uni_is_L('A') && uni_is_L('z') && uni_is_L(0x4E00) && uni_is_L(0xAC00));
        CHECK(!uni_is_L('0') && !uni_is_L(' ') && !uni_is_L(0x3000));
        CHECK(uni_is_N('0') && uni_is_N(0xFF10) && !uni_is_N('A'));
        CHECK(uni_is_S(' ') && uni_is_S(0x3000) && !uni_is_S('A'));
        CHECK(uni_is_Lu('A') && uni_is_Lu(0x0410) && !uni_is_Lu('a'));
        CHECK(uni_is_Ll('a') && uni_is_Ll(0x0430) && !uni_is_Ll('A'));
        CHECK(uni_is_L('A') && uni_is_L('a'));
        CHECK(uni_to_lower('A') == 'a' && uni_to_lower(0x0410) == 0x0430);
        CHECK(uni_to_lower('a') == 'a' && uni_to_lower('0') == '0');
        CHECK(uni_to_upper('a') == 'A' && uni_to_upper(0x0430) == 0x0410);
        CHECK(uni_to_upper('A') == 'A' && uni_to_upper('0') == '0');
        CHECK(uni_is_M(0x0301) && uni_is_M(0x20D0) && uni_is_M(0xFE20));
        CHECK(!uni_is_M('A') && !uni_is_M('0') && !uni_is_M(0x4E00));
        CHECK(uni_is_P('!') && uni_is_P(0x2014) && uni_is_P(0x3001) && uni_is_P(0xFF01));
        CHECK(!uni_is_P('A') && !uni_is_P('0') && !uni_is_P(' ') && !uni_is_P(0x3000));
        std::string det = detokenize_response("hi\"x");
        CHECK(det.find("\"text\":\"") != std::string::npos);
        CHECK(det.find("hi") != std::string::npos);
        std::string tokj = tokenize_response({1, 2, 3});
        CHECK(tokj.find("\"count\":3") != std::string::npos);
        CHECK(tokj.find("1,2,3") != std::string::npos);
        std::string cnt = count_response(3);
        CHECK(cnt == "{\"count\":3}");
        CHECK(cnt.find("tokens") == std::string::npos);
        CHECK(count_response(0) == "{\"count\":0}");
        CHECK(ready_json(true) ==
              "{\"ready\":true,\"created\":" + std::to_string(serve_created()) + "}");
        CHECK(ready_json(false) ==
              "{\"ready\":false,\"created\":" + std::to_string(serve_created()) + "}");
        CHECK(version_json().find("\"name\":\"micro-vllm\"") != std::string::npos);
        CHECK(version_json().find("\"engine\":\"host\"") != std::string::npos);
        CHECK(version_json().find("\"created\":" + std::to_string(serve_created())) !=
              std::string::npos);
        CHECK(mux_format_accept(9, 4) == "ACCEPT 9 4\n");
        CHECK(mux_format_error(3, "EMPTY_PROMPT") == "ERROR 3 EMPTY_PROMPT\n");
        CHECK(mux_format_error(1, nullptr) == "ERROR 1 BAD_FRAME\n");
        CHECK(mux_format_error(1, "") == "ERROR 1 BAD_FRAME\n");
        std::string ctok = anthropic_count_tokens_response(7);
        CHECK(ctok.find("\"type\":\"message_count_tokens_response\"") != std::string::npos);
        CHECK(ctok.find("\"input_tokens\":7") != std::string::npos);
        CHECK(anthropic_count_tokens_response(-3).find("\"input_tokens\":0") != std::string::npos);
        {
            LegacyCommand lc;
            std::string lerr;
            CHECK(legacy_parse_line("\x02RESET", lc, lerr) && lc.kind == LegacyCmd::Reset);
            CHECK(legacy_parse_line("\x02MORE", lc, lerr) && lc.kind == LegacyCmd::More);
            CHECK(legacy_parse_line("\x02PROMPT 4 16 0.7 0.9 2", lc, lerr));
            CHECK(lc.kind == LegacyCmd::Prompt && lc.payload_bytes == 4 && lc.max_tokens == 16);
            CHECK_NEAR(lc.temperature, 0.7f, 1e-6);
            CHECK_NEAR(lc.top_p, 0.9f, 1e-6);
            CHECK(lc.have_slot && lc.slot == 2);
            CHECK(legacy_parse_line("hello", lc, lerr) && lc.kind == LegacyCmd::Line);
            CHECK(!legacy_parse_line("", lc, lerr) && lerr == "empty");
            CHECK(!legacy_parse_line("\x02PROMPT 4 0 0.7 0.9", lc, lerr) && lerr == "bad prompt");
            CHECK(legacy_format_ready() == "\x01\x01READY\x01\x01\n");
            CHECK(legacy_format_end() == "\x01\x01END\x01\x01\n");
            CHECK(legacy_format_stat(3, 1.5, 50.0, 1.25) == "STAT 3 1.50 50.0 1.25\n");
        }
        std::string ping = anthropic_sse_ping();
        CHECK(ping.find("event: ping") != std::string::npos);
        CHECK(ping.find("\"type\":\"ping\"") != std::string::npos);
    }
    {
        using namespace mvllm;
        const float lo[] = {0.f, 3.f, 1.f, 2.f};
        LogitTop top[5];
        CHECK(logit_topn(lo, 4, top, 5) == 4);
        CHECK(top[0].id == 1 && top[0].logit == 3.f);
        CHECK(top[1].id == 3 && top[1].logit == 2.f);
        CHECK(top[4].id == -1);
        std::string dump = logit_dump_line(lo, 4);
        CHECK(dump.find("[LOGITS] 1:3.000000 3:2.000000 2:1.000000 0:0.000000") == 0);
        CHECK(dump.back() == '\n');
        std::string gap = logit_gap_line(0, lo, 4);
        CHECK(gap.find("LOGITGAP pos=0 top1=1:3.000000 top2=3:2.000000 gap=1.") == 0);
        CHECK(logit_dump_line(nullptr, 0) == "[LOGITS]\n");
    }
    {
        using namespace mvllm;
        GenParams gp;
        std::string merr;
        CHECK(mux_apply_extra_json(
            "{\"persist\":\"/tmp/a.coli_kv\",\"persist_ver\":2,\"prefix_bytes\":4,"
            "\"prefix_reuse\":3,\"stop_ids\":[7,9],\"eos_only\":true}",
            gp, merr));
        CHECK(gp.persist_path == "/tmp/a.coli_kv");
        CHECK(gp.persist_ver == 2);
        CHECK(gp.prefix_bytes == 4 && gp.prefix_reuse == 3);
        CHECK(gp.stop_ids.size() == 2 && gp.stop_ids[0] == 7 && gp.stop_ids[1] == 9);
        CHECK(gp.eos_only);
        CHECK(mux_apply_extra_json("{\"coli_kv\":\"/tmp/b.coli_kv\",\"kv_ver\":3}", gp, merr));
        CHECK(gp.persist_path == "/tmp/b.coli_kv" && gp.persist_ver == 3);
        CHECK(!mux_apply_extra_json("{\"persist_ver\":9}", gp, merr));
        CHECK(!mux_apply_extra_json("{\"stop_ids\":1}", gp, merr));
        CHECK(!mux_apply_extra_json("{\"eos_only\":1}", gp, merr));
        CHECK(mux_apply_extra_json("{\"stop_ids\":[]}", gp, merr) && gp.stop_ids.empty());
        CHECK(mux_apply_extra_json("{\"tool_choice\":\"none\"}", gp, merr));
        CHECK(gp.tool_choice == "none");
        CHECK(mux_apply_extra_json("{\"function_call\":\"auto\",\"tool_choice\":\"required\"}",
                                  gp, merr));
        CHECK(gp.tool_choice == "required");
        CHECK(!mux_apply_extra_json("{\"tool_choice\":\"\"}", gp, merr));
        CHECK(!mux_apply_extra_json("{\"function_call\":1}", gp, merr));
        CHECK(mux_apply_extra_json("{\"reasoning_effort\":\"high\"}", gp, merr));
        CHECK(gp.reasoning_effort == "high" && gp.think);
        CHECK(mux_apply_extra_json(
            "{\"reasoning-effort\":\"high\",\"reasoning_effort\":\"none\"}", gp, merr));
        CHECK(gp.reasoning_effort == "none" && !gp.think);
        CHECK(!mux_apply_extra_json("{\"reasoning_effort\":\"max\"}", gp, merr));
        CHECK(!mux_apply_extra_json("{\"reasoning_effort\":1}", gp, merr));
        ModelConfig cfg;
        cfg.eos = 1;
        GenParams sp;
        sp.stop_ids = {5};
        CHECK(gen_stop_id(5, cfg, sp) && gen_stop_id(1, cfg, sp) && !gen_stop_id(2, cfg, sp));
        float lo[] = {0.f, 3.f, 1.f, 2.f};
        uint8_t allow[4] = {1, 0, 1, 1};
        uint64_t rng = 1;
        CHECK(sample_token(lo, 4, 0.f, 1.f, &rng, allow) == 3);
        const char *old_kv = std::getenv("MVLLM_KV");
        std::string old_kv_s = old_kv ? old_kv : "";
        setenv("MVLLM_KV", "/tmp/wired.coli_kv", 1);
        RuntimeConfig rtenv = runtime_from_env();
        CHECK(rtenv.kv_path == "/tmp/wired.coli_kv");
        if (!old_kv_s.empty())
            setenv("MVLLM_KV", old_kv_s.c_str(), 1);
        else
            unsetenv("MVLLM_KV");
        const char *old_mq = std::getenv("COLI_MAX_QUEUE");
        std::string old_mq_s = old_mq ? old_mq : "";
        const char *old_qt = std::getenv("COLI_QUEUE_TIMEOUT");
        std::string old_qt_s = old_qt ? old_qt : "";
        setenv("COLI_MAX_QUEUE", "16", 1);
        setenv("COLI_QUEUE_TIMEOUT", "60", 1);
        RuntimeConfig rtq = runtime_from_env();
        CHECK(rtq.max_queue == 16 && rtq.queue_timeout_s == 60);
        if (!old_mq_s.empty())
            setenv("COLI_MAX_QUEUE", old_mq_s.c_str(), 1);
        else
            unsetenv("COLI_MAX_QUEUE");
        if (!old_qt_s.empty())
            setenv("COLI_QUEUE_TIMEOUT", old_qt_s.c_str(), 1);
        else
            unsetenv("COLI_QUEUE_TIMEOUT");
        const std::string kdir = tmpdir();
        write_file(kdir + "/config.json", R"({
          "model_type": "kimi_linear",
          "architectures": ["KimiLinearForCausalLM"],
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "vocab_size": 32,
          "first_k_dense_replace": 1,
          "intermediate_size": 32,
          "num_attention_heads": 4,
          "q_lora_rank": 8,
          "kv_lora_rank": 8,
          "qk_nope_head_dim": 8,
          "v_head_dim": 8,
          "num_experts": 2,
          "num_experts_per_token": 1,
          "moe_intermediate_size": 16,
          "routed_expert_hidden_size": 16,
          "num_shared_experts": 1,
          "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "kda_layers": [1],
            "full_attn_layers": [2],
            "use_full_rank_gate": true
          },
          "bos_token_id": 0,
          "eos_token_id": 1
        })");
        Engine ek;
        RuntimeConfig rt;
        rt.expert_gb = 0.001;
        std::string err;
        CHECK(ek.load(kdir, rt, err) == Status::Ok);
        const std::string kvpath = kdir + "/sess.coli_kv";
        CHECK(ek.persist_open(kvpath, 1, err) == Status::Ok);
        CHECK(ek.persist_nrec() == 0);
        GenParams gp2;
        gp2.max_new_tokens = 2;
        gp2.eos = 1;
        gp2.stop_ids = {31};
        gp2.persist_path = kvpath;
        gp2.persist_ver = 1;
        GenResult gr;
        CHECK(ek.generate("hi", gp2, gr, err) == Status::Ok);
        CHECK(ek.persist_nrec() > 0);
        CHECK(ek.prefix_reuse_len(0) == ek.persist_nrec());
        CHECK(!ek.persist_hist().empty());
        const std::vector<int> hist1 = ek.persist_hist();
        ek.persist_close();
        CHECK(ek.persist_nrec() == 0);
        CHECK(ek.persist_open(kvpath, 1, err) == Status::Ok);
        CHECK(ek.persist_nrec() == static_cast<int>(hist1.size()));
        CHECK(ek.persist_hist() == hist1);
        CHECK(ek.prefix_reuse_len(0) == static_cast<int>(hist1.size()));
        KvPersist kp;
        KvPersistConfig pcfg;
        pcfg.n_layers = ek.config().n_layers > 0 ? ek.config().n_layers : 1;
        pcfg.kv_lora = std::max(ek.config().mla.kv_lora, 0);
        pcfg.qk_rope = std::max(ek.config().mla.qk_rope, 0);
        pcfg.index_hd = std::max(ek.config().dsa.head_dim, 0);
        pcfg.vocab = std::max(ek.config().vocab, 0);
        pcfg.has_index.assign(static_cast<size_t>(pcfg.n_layers), 0);
        CHECK(kp.open(kvpath, pcfg, err) == Status::Ok);
        std::vector<int> phist;
        std::vector<KvPersistRecord> prows;
        CHECK(kp.load(phist, &prows, err) == ek.persist_nrec());
        CHECK(!prows.empty());
        bool any_l = false;
        for (const auto &r : prows) {
            for (float v : r.L) {
                if (v != 0.f) {
                    any_l = true;
                    break;
                }
            }
            if (any_l)
                break;
        }
        CHECK(any_l);
        FamilyEngine *fe = ek.family_impl();
        CHECK(fe != nullptr);
        KvPersistRecord one;
        one.L.assign(static_cast<size_t>(pcfg.n_layers) * static_cast<size_t>(pcfg.kv_lora), 0.f);
        one.R.assign(static_cast<size_t>(pcfg.n_layers) * static_cast<size_t>(pcfg.qk_rope), 0.f);
        CHECK(fe->export_kv_rows(0, 0, 1, &one) == 1);
        GenParams extra;
        std::string xerr;
        CHECK(mux_apply_extra_json("{\"prefix_bytes\":2,\"prefix_reuse\":0}", extra, xerr));
        CHECK(extra.prefix_bytes == 2 && extra.prefix_reuse == 0);
        const std::string gdir = tmpdir();
        write_file(gdir + "/config.json", R"({
          "model_type": "glm",
          "architectures": ["Glm5ForConditionalGeneration"],
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "vocab_size": 32,
          "first_k_dense_replace": 1,
          "intermediate_size": 32,
          "num_experts": 2,
          "num_experts_per_token": 1,
          "moe_intermediate_size": 16,
          "num_shared_experts": 1,
          "q_lora_rank": 8,
          "kv_lora_rank": 8,
          "qk_nope_head_dim": 8,
          "layer_types": ["linear","full_attention"],
          "linear_attn_config": {"num_heads": 2, "head_dim": 8, "short_conv_kernel_size": 4},
          "bos_token_id": 0,
          "eos_token_id": 1
        })");
        Engine eg;
        CHECK(eg.load(gdir, rt, err) == Status::Ok);
        GenParams gpg;
        gpg.max_new_tokens = 2;
        gpg.eos = 1;
        GenResult grg;
        CHECK(eg.generate("ok", gpg, grg, err) == Status::Ok);
        FamilyEngine *fg = eg.family_impl();
        CHECK(fg != nullptr);
        KvPersistRecord grow;
        const int gl = std::max(eg.config().n_layers, 1);
        const int gkv = std::max(eg.config().mla.kv_lora, 0);
        const int gqr = std::max(eg.config().mla.qk_rope, 0);
        grow.L.assign(static_cast<size_t>(gl) * static_cast<size_t>(gkv), 0.f);
        grow.R.assign(static_cast<size_t>(gl) * static_cast<size_t>(gqr), 0.f);
        grow.I.assign(static_cast<size_t>(std::max(eg.config().dsa.head_dim, 0)), 0.f);
        CHECK(fg->export_kv_rows(0, 0, 1, &grow) == 1);
        KvPersistRecord restored;
        restored.L.assign(one.L.size(), 0.f);
        restored.R.assign(one.R.size(), 0.f);
        ek.persist_close();
        CHECK(ek.persist_open(kvpath, 1, err) == Status::Ok);
        CHECK(fe->export_kv_rows(0, 0, 1, &restored) == 1);
        CHECK(restored.L == one.L);
        CHECK(restored.R == one.R);
        std::vector<int> ids;
        CHECK(extract_json_int_array("{\"stop_ids\":[7,9,0]}", "stop_ids", ids));
        CHECK(ids.size() == 3 && ids[0] == 7 && ids[2] == 0);
        CHECK(extract_json_int_array("{\"stop_ids\":[]}", "stop_ids", ids) && ids.empty());
        CHECK(!extract_json_int_array("{\"stop_ids\":1}", "stop_ids", ids));
        CHECK(!extract_json_int_array("{}", "stop_ids", ids));
    }
    {
        using namespace mvllm;
        std::vector<ChatMessage> msgs;
        GenParams gp;
        std::string err;
        CHECK(anthropic_to_chat(
            "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            "\"persist\":\"/tmp/a.coli_kv\",\"persist_ver\":2,\"prefix_bytes\":4,"
            "\"stop_ids\":[3,5],\"eos_only\":true}",
            msgs, gp, err));
        CHECK(gp.persist_path == "/tmp/a.coli_kv");
        CHECK(gp.persist_ver == 2);
        CHECK(gp.prefix_bytes == 4);
        CHECK(gp.stop_ids.size() == 2 && gp.stop_ids[0] == 3 && gp.stop_ids[1] == 5);
        CHECK(gp.eos_only);
        char a0[] = "micro-vllm";
        char a1[] = "--persist";
        char a2[] = "/tmp/b.coli_kv";
        char a3[] = "--persist-ver";
        char a4[] = "3";
        char a5[] = "--stop-id";
        char a6[] = "9";
        char a7[] = "--eos-only";
        char a8[] = "--prefix-bytes";
        char a9[] = "6";
        char *av[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9};
        GenParams cgp;
        CliGenExtras cex;
        std::string cerr;
        CHECK(apply_cli_gen_flags(10, av, cgp, cex, cerr));
        CHECK(cgp.persist_path == "/tmp/b.coli_kv");
        CHECK(cgp.persist_ver == 3);
        CHECK(cgp.stop_ids.size() == 1 && cgp.stop_ids[0] == 9);
        CHECK(cgp.eos_only && cgp.prefix_bytes == 6);
        char e0[] = "micro-vllm";
        char e1[] = "--eos-only";
        char e2[] = "--no-eos-only";
        char *ev[] = {e0, e1, e2};
        CHECK(apply_cli_gen_flags(3, ev, cgp, cex, cerr));
        CHECK(!cgp.eos_only);
        char e3[] = "micro-vllm";
        char e4[] = "--no-eos-only";
        char e5[] = "--eos-only";
        char *ew[] = {e3, e4, e5};
        CHECK(apply_cli_gen_flags(3, ew, cgp, cex, cerr));
        CHECK(cgp.eos_only);
        char b0[] = "x";
        char b1[] = "--persist-ver";
        char b2[] = "9";
        char *bv[] = {b0, b1, b2};
        CHECK(!apply_cli_gen_flags(3, bv, cgp, cex, cerr));
        const std::string kdir = tmpdir();
        write_file(kdir + "/config.json", R"({
          "model_type": "kimi_linear",
          "architectures": ["KimiLinearForCausalLM"],
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "vocab_size": 32,
          "first_k_dense_replace": 1,
          "intermediate_size": 32,
          "num_attention_heads": 4,
          "q_lora_rank": 8,
          "kv_lora_rank": 8,
          "qk_nope_head_dim": 8,
          "v_head_dim": 8,
          "num_experts": 2,
          "num_experts_per_token": 1,
          "moe_intermediate_size": 16,
          "routed_expert_hidden_size": 16,
          "num_shared_experts": 1,
          "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "kda_layers": [1],
            "full_attn_layers": [2],
            "use_full_rank_gate": true
          },
          "bos_token_id": 0,
          "eos_token_id": 1
        })");
        Engine ek;
        RuntimeConfig rt;
        rt.expert_gb = 0.001;
        CHECK(ek.load(kdir, rt, err) == Status::Ok);
        CHECK(ek.prefix_match(0, {1, 2, 3}) == 0);
        ek.prefix_commit(0, {1, 2, 3});
        CHECK(ek.prefix_match(0, {1, 2, 3, 4}) == 3);
        CHECK(ek.prefix_match(0, {1, 2}) == 0);
        const std::string kvpath = kdir + "/sched.coli_kv";
        CHECK(ek.persist_open(kvpath, 1, err) == Status::Ok);
        CHECK(ek.persist_commit(0, {1, 2, 3}, err) == Status::Ok);
        CHECK(ek.persist_nrec() == 3);
        ek.prefix_commit(0, {1, 2, 3});
        BatchScheduler &sch = ek.scheduler();
        sch.configure(4, 10);
        std::string serr;
        uint64_t sid = sch.submit(0, {1, 2, 3, 4}, {}, serr);
        CHECK(sid != 0);
        const BatchJob *job = sch.job(sid);
        CHECK(job && job->reuse == 3);
        sch.cancel(sid);
    }
    {
        using namespace mvllm;
        DsaConfig d;
        d.n_heads = 1;
        d.head_dim = 2;
        d.topk = 2;
        d.kpool = 2;
        d.always_select_tail = true;
        float q[2] = {1.f, 0.f};
        float keys[10] = {1.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.5f, 0.5f};
        float gates[10] = {};
        float hw[1] = {1.f};
        int sel[8];
        CHECK(dsa_select(sel, q, keys, gates, hw, nullptr, 5, d) == 3);
        int ranged[8];
        CHECK(dsa_select_range(ranged, q, keys, gates, hw, nullptr, nullptr, 5, d, 4, 5) == 0);
        CHECK(ranged[0] == sel[0] && ranged[1] == sel[1] && ranged[2] == sel[2]);
        CHECK(dsa_select_range(ranged, q, keys, gates, hw, nullptr, nullptr, 5, d, -1, 5) == -1);
        const std::string kdir = tmpdir();
        write_file(kdir + "/config.json", R"({
          "model_type": "kimi_linear",
          "architectures": ["KimiLinearForCausalLM"],
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "vocab_size": 32,
          "first_k_dense_replace": 1,
          "intermediate_size": 32,
          "num_attention_heads": 4,
          "q_lora_rank": 8,
          "kv_lora_rank": 8,
          "qk_nope_head_dim": 8,
          "v_head_dim": 8,
          "num_experts": 2,
          "num_experts_per_token": 1,
          "moe_intermediate_size": 16,
          "routed_expert_hidden_size": 16,
          "num_shared_experts": 1,
          "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "kda_layers": [1],
            "full_attn_layers": [2],
            "use_full_rank_gate": true
          },
          "bos_token_id": 0,
          "eos_token_id": 1
        })");
        Engine ek;
        RuntimeConfig rt;
        rt.expert_gb = 0.001;
        std::string err;
        CHECK(ek.load(kdir, rt, err) == Status::Ok);
        ek.prefix_commit(1, {9, 8, 7});
        CHECK(ek.prefix_match(1, {9, 8, 7, 6}) == 3);
        const std::string emptykv = kdir + "/empty.coli_kv";
        CHECK(ek.persist_open(emptykv, 1, err, 0) == Status::Ok);
        CHECK(ek.prefix_match(1, {9, 8, 7, 6}) == 3);
        ek.prefix_commit(0, {1, 2});
        CHECK(ek.persist_open(emptykv, 1, err, 0) == Status::Ok);
        CHECK(ek.prefix_match(0, {1, 2, 3}) == 2);
        const std::string ldir = tmpdir();
        write_file(ldir + "/config.json", R"({
          "model_type":"llama","architectures":["LlamaForCausalLM"],
          "hidden_size":32,"num_hidden_layers":1,"vocab_size":16,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
          "intermediate_size":64
        })");
        Engine el;
        CHECK(el.load(ldir, rt, err) == Status::Ok);
        GenParams lgp;
        lgp.max_new_tokens = 2;
        lgp.apply_template = false;
        lgp.eos = 1;
        GenResult lgr;
        CHECK(el.generate("hi", lgp, lgr, err) == Status::Ok);
        const std::string lkv = ldir + "/llama.coli_kv";
        CHECK(el.persist_open(lkv, 1, err, 0) == Status::Ok);
        lgp.persist_path = lkv;
        lgp.persist_ver = 1;
        CHECK(el.generate("hi", lgp, lgr, err) == Status::Ok);
        CHECK(el.persist_nrec() > 0);
        FamilyEngine *fl = el.family_impl();
        CHECK(fl != nullptr);
        KvPersistRecord rec;
        rec.L.assign(static_cast<size_t>(std::max(el.config().n_kv_heads, 1) *
                                         std::max(el.config().head_dim, 1)),
                     0.f);
        rec.R = rec.L;
        CHECK(fl->export_kv_rows(0, 0, 1, &rec) == 1);
    }
    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail ? 1 : 0;
}
