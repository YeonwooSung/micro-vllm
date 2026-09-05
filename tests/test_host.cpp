#include "core/config.hpp"
#include "engine.hpp"
#include "gpu/backend.hpp"
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
#include "model/h3_text.hpp"
#include "model/h3_audio_vae.hpp"
#include "model/h3_layout.hpp"
#include "model/h3_vision.hpp"
#include "model/h3_mm.hpp"
#include "model/h3_dit_schedule.hpp"
#include "model/h3_canvas.hpp"
#include "model/h3_adaln.hpp"
#include "store/block_store.hpp"
#include "store/expert_store.hpp"
#include "store/kv_persist.hpp"
#include "store/route_usage.hpp"
#include "tok/tokenizer.hpp"
#include "tok/k3_tools.hpp"
#include "tok/gbnf.hpp"
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
#include "serve/cli_flags.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
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
    store.close();

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
    CHECK(hr.blocks_streamed == 8); // 2 evals * 4 layers
    CHECK(!hr.output_path.empty());
    CHECK(hr.audio_used);
    CHECK(hr.note.find("audio=") != std::string::npos);
    CHECK(hr.note.find("pack=text+audio+video") != std::string::npos);
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
    ExpertStoreStats st{};
    ek.expert_stats(st);
    CHECK(st.requests > 0);
    CHECK(st.misses + st.hits == st.requests);
    CHECK(gr.completion_tokens > 0);

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
    ExpertStoreStats gst{};
    eg.expert_stats(gst);
    CHECK(gst.requests > 0);
    CHECK(gst.misses + gst.hits == gst.requests);
    CHECK(gr.completion_tokens > 0);

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
    test_offload_generate();
    test_glm53_container();
    test_k3_mxfp4_container();
    test_h3_checkpoint();
    test_kda_short_conv();
    test_dsa();
    test_shard_probe();
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
        CHECK(ss.try_acquire(1));
        CHECK(ss.busy(1));
        CHECK(!ss.try_acquire(1));
        ss.release(1);
        CHECK(ss.try_acquire(1));
        ss.commit(1, {10, 11, 12});
        CHECK(ss.match(1, {10, 11, 99}) == 2);
        ss.reset(1);
        CHECK(ss.match(1, {10, 11}) == 0);
        CHECK(!ss.busy(1));
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
        }
        std::string done = mux_format_done(7, 3, 10, 0, 2);
        CHECK(done.find("DONE 7 STAT 3") != std::string::npos);
        CHECK(done.find("10 0 2") != std::string::npos);
        std::string rtoks =
            openai_chat_response("id1", "kimi", "hello", 3, 2, "plan", "stop", 0, 4);
        CHECK(rtoks.find("\"reasoning_tokens\":4") != std::string::npos);
        CHECK(rtoks.find("completion_tokens_details") != std::string::npos);
        std::string nor =
            openai_chat_response("id1", "kimi", "hello", 3, 2);
        CHECK(nor.find("reasoning_tokens") == std::string::npos);
        std::string hz = health_json(nullptr);
        CHECK(hz.find("\"ok\":true") != std::string::npos);
        CHECK(hz.find("kv_slots") != std::string::npos);
        CHECK(hz.find("\"model\"") == std::string::npos);
        std::string mo = openai_model_object("glm53");
        CHECK(mo.find("\"created\":0") != std::string::npos);
        CHECK(mo.find("glm53") != std::string::npos);
        CHECK(openai_models_response("glm53").find("\"created\":0") != std::string::npos);
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
        std::string st = mux_format_stat(2);
        CHECK(st.find("STAT 2 0.00 0.0 0.00") != std::string::npos);
        std::string td = anthropic_sse_thinking_delta("plan");
        CHECK(td.find("event: content_block_delta") != std::string::npos);
        CHECK(td.find("thinking_delta") != std::string::npos);
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
        std::string r, c;
        split_assistant_text(Family::Llama, "<think>plan</think>Hello<|eot_id|>\n", r, c);
        CHECK(r == "plan");
        CHECK(c == "Hello");
        BatchScheduler sch;
        sch.configure(4, 10);
        CHECK(sch.max_queue() == 4);
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
        std::string perf = mux_format_perf(9, 0.5, 0.1, 0, 0, 0, 0, 0);
        CHECK(perf.find("PERF 9 ") == 0);
        CHECK(perf.back() == '\n');
        float ent[2] = {1.5f, 2.f};
        CHECK(mux_format_entropy(ent, 2) == "ENTROPY 1.5 2\n");
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
    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail ? 1 : 0;
}
