#include "core/config.hpp"
#include "engine.hpp"
#include "io/file_io.hpp"
#include "io/safetensors.hpp"
#include "model/h3_vae.hpp"
#include "quant/quant.hpp"
#include "quant/weight.hpp"
#include "serve/http_server.hpp"
#include "store/block_store.hpp"
#include "store/expert_store.hpp"
#include "tok/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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

    std::string glm = tk.apply_chat(Family::Glm53, {{"user", "hi"}}, false);
    CHECK(glm.find("[gMASK]<sop>") == 0);
    CHECK(glm.find("<|user|>hi") != std::string::npos);
    CHECK(glm.find("<|assistant|><think></think>") != std::string::npos);
    std::string glmt = tk.apply_chat(Family::Glm53, {{"user", "hi"}}, true);
    CHECK(glmt.find("<|assistant|><think>") != std::string::npos);
    CHECK(glmt.find("<think></think>") == std::string::npos ||
          glmt.rfind("<|assistant|><think>") > glmt.find("<|user|>"));
    std::string k3 = tk.apply_chat(Family::KimiK3, {{"user", "hi"}}, false);
    CHECK(k3.find("<|im_start|>user") != std::string::npos);
    CHECK(k3.find("<|im_start|>assistant") != std::string::npos);
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
    hp.output_path = hdir + "/out.txt";
    H3GenResult hr;
    CHECK(eh.generate_video(hp, hr, err) == Status::Ok);
    CHECK(hr.blocks_streamed == 8); // 2 evals * 4 layers
    CHECK(!hr.output_path.empty());
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
    CHECK(hr.blocks_streamed == 4);
    CHECK(hr.vae_used);
    CHECK(hr.frames == 5);
    CHECK(hr.width == 32 && hr.height == 32);
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
    test_config_and_families();
    test_offload_generate();
    test_glm53_container();
    test_k3_mxfp4_container();
    test_h3_checkpoint();
    test_kda_short_conv();
    test_dsa();
    test_moe_union();
    test_dense_bits_generate();
    test_mla_absorb();
    test_mla_generate();
    test_h3_vae();
    std::cout << "passed=" << g_pass << " failed=" << g_fail << "\n";
    return g_fail ? 1 : 0;
}
