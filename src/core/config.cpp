#include "config.hpp"

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace mvllm {
namespace {

using json = nlohmann::json;

std::string read_file(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int jnum(const json &o, const char *k, int fallback) {
    if (!o.contains(k) || o[k].is_null())
        return fallback;
    if (o[k].is_number_integer())
        return o[k].get<int>();
    if (o[k].is_number())
        return static_cast<int>(o[k].get<double>());
    return fallback;
}

float jfloat(const json &o, const char *k, float fallback) {
    if (!o.contains(k) || o[k].is_null())
        return fallback;
    if (o[k].is_number())
        return static_cast<float>(o[k].get<double>());
    return fallback;
}

std::string jstr(const json &o, const char *k, const std::string &fallback = {}) {
    if (!o.contains(k) || !o[k].is_string())
        return fallback;
    return o[k].get<std::string>();
}

const json &text_object(const json &root) {
    if (root.contains("text_config") && root["text_config"].is_object())
        return root["text_config"];
    return root;
}

} // namespace

void apply_family_defaults(ModelConfig &cfg) {
    // Full-size constants only apply when the checkpoint omitted geometry
    // (or is the released model). Tiny fixtures must keep their own ranks.
    const bool full = cfg.hidden == 0 || cfg.hidden >= 1024;
    switch (cfg.family) {
    case Family::KimiK3:
        if (!cfg.hidden)
            cfg.hidden = 7168;
        if (!cfg.n_layers)
            cfg.n_layers = 93;
        if (!cfg.moe.n_experts && full)
            cfg.moe.n_experts = 896;
        if (!cfg.moe.topk)
            cfg.moe.topk = 16;
        if (!cfg.moe.intermediate)
            cfg.moe.intermediate = 3072;
        if (!cfg.moe.latent)
            cfg.moe.latent = 3584;
        if (!cfg.moe.n_shared)
            cfg.moe.n_shared = 2;
        cfg.moe.situ_b1 = cfg.moe.situ_b1 ? cfg.moe.situ_b1 : 4.f;
        cfg.moe.situ_b2 = cfg.moe.situ_b2 ? cfg.moe.situ_b2 : 25.f;
        if (!cfg.kda.heads)
            cfg.kda.heads = full ? 96 : 2;
        if (!cfg.kda.head_dim)
            cfg.kda.head_dim = full ? 128 : 16;
        if (!cfg.kda.conv_k)
            cfg.kda.conv_k = 4;
        cfg.kda.full_rank_gate = true;
        if (!cfg.mla.n_heads)
            cfg.mla.n_heads = full ? 96 : 4;
        if (!cfg.mla.q_lora && full)
            cfg.mla.q_lora = 1536;
        if (!cfg.mla.kv_lora && full)
            cfg.mla.kv_lora = 512;
        if (!cfg.mla.qk_nope && full)
            cfg.mla.qk_nope = 128;
        if (!cfg.mla.v_head && full)
            cfg.mla.v_head = 128;
        if (!cfg.mla.qk_rope && full)
            cfg.mla.qk_rope = 64;
        cfg.mla.nope = true;
        cfg.mla.output_gate = true;
        if (!cfg.attn_res.block_size)
            cfg.attn_res.block_size = 12;
        break;
    case Family::Glm53:
        if (!cfg.hidden)
            cfg.hidden = 4096;
        if (!cfg.n_layers)
            cfg.n_layers = 45;
        if (!cfg.vocab && full)
            cfg.vocab = 154880;
        if (!cfg.first_dense && full)
            cfg.first_dense = 3;
        if (!cfg.dense_intermediate && full)
            cfg.dense_intermediate = 12288;
        if (!cfg.moe.n_experts && full)
            cfg.moe.n_experts = 288;
        if (!cfg.moe.topk)
            cfg.moe.topk = 8;
        if (!cfg.moe.intermediate)
            cfg.moe.intermediate = 2048;
        if (!cfg.moe.n_shared)
            cfg.moe.n_shared = 1;
        if (cfg.moe.routed_scale == 1.f)
            cfg.moe.routed_scale = 2.5f;
        if (!cfg.mla.q_lora && full)
            cfg.mla.q_lora = 1536;
        if (!cfg.mla.kv_lora && full)
            cfg.mla.kv_lora = 512;
        if (!cfg.mla.qk_nope && full)
            cfg.mla.qk_nope = 256;
        if (!cfg.mla.qk_rope)
            cfg.mla.qk_rope = 0;
        cfg.mla.nope = cfg.mla.qk_rope == 0;
        cfg.mla.output_gate = false;
        if (!cfg.mla.n_heads && full)
            cfg.mla.n_heads = 96;
        if (!cfg.mla.v_head && full)
            cfg.mla.v_head = 256;
        if (!cfg.mhc.mult)
            cfg.mhc.mult = 4;
        if (!cfg.mhc.iters)
            cfg.mhc.iters = 20;
        if (!cfg.dsa.kpool)
            cfg.dsa.kpool = 1;
        cfg.kda.full_rank_gate = false;
        if (!cfg.kda.conv_k)
            cfg.kda.conv_k = 4;
        break;
    case Family::H3:
        if (!cfg.h3.dit_layers)
            cfg.h3.dit_layers = 50;
        if (!cfg.h3.default_width)
            cfg.h3.default_width = 864;
        if (!cfg.h3.default_height)
            cfg.h3.default_height = 480;
        if (!cfg.h3.stream_slots)
            cfg.h3.stream_slots = 2;
        break;
    case Family::Llama:
        if (!cfg.hidden)
            cfg.hidden = 2048;
        if (!cfg.n_layers)
            cfg.n_layers = 16;
        if (!cfg.vocab)
            cfg.vocab = 128256;
        if (!cfg.n_q_heads)
            cfg.n_q_heads = 32;
        if (!cfg.n_kv_heads)
            cfg.n_kv_heads = 8;
        if (!cfg.head_dim)
            cfg.head_dim = 64;
        if (!cfg.dense_intermediate)
            cfg.dense_intermediate = 8192;
        break;
    case Family::Dsv4:
        if (!cfg.hidden)
            cfg.hidden = 4096;
        if (!cfg.n_layers)
            cfg.n_layers = 43;
        if (!cfg.vocab && full)
            cfg.vocab = 129280;
        if (!cfg.moe.n_experts && full)
            cfg.moe.n_experts = 256;
        if (!cfg.moe.topk)
            cfg.moe.topk = 6;
        if (!cfg.moe.intermediate && full)
            cfg.moe.intermediate = 2048;
        if (!cfg.moe.n_shared)
            cfg.moe.n_shared = 1;
        if (cfg.moe.routed_scale == 1.f)
            cfg.moe.routed_scale = 1.5f;
        if (!cfg.moe.swiglu_limit)
            cfg.moe.swiglu_limit = 10.f;
        if (!cfg.mla.n_heads)
            cfg.mla.n_heads = full ? 64 : (cfg.n_q_heads > 0 ? cfg.n_q_heads : 2);
        if (!cfg.head_dim)
            cfg.head_dim = full ? 512 : 0;
        if (!cfg.mla.q_lora && full)
            cfg.mla.q_lora = 1024;
        if (!cfg.mla.qk_rope && full)
            cfg.mla.qk_rope = 64;
        if (!cfg.mla.qk_nope && cfg.head_dim > cfg.mla.qk_rope)
            cfg.mla.qk_nope = cfg.head_dim - cfg.mla.qk_rope;
        if (!cfg.mla.v_head)
            cfg.mla.v_head = cfg.head_dim;
        if (!cfg.mla.kv_lora)
            cfg.mla.kv_lora = cfg.head_dim;
        if (!cfg.o_lora && full)
            cfg.o_lora = 1024;
        if (!cfg.o_groups && full)
            cfg.o_groups = 8;
        if (!cfg.sliding_window && full)
            cfg.sliding_window = 128;
        if (!cfg.dsa.topk && full)
            cfg.dsa.topk = 512;
        if (!cfg.dsa.n_heads && full)
            cfg.dsa.n_heads = 64;
        if (!cfg.dsa.head_dim && full)
            cfg.dsa.head_dim = 128;
        if (!cfg.mhc.mult)
            cfg.mhc.mult = 4;
        if (!cfg.mhc.iters)
            cfg.mhc.iters = 20;
        cfg.mla.nope = cfg.mla.qk_rope == 0;
        cfg.mla.output_gate = false;
        if (cfg.first_dense >= cfg.n_layers && cfg.n_layers > 0)
            cfg.first_dense = 0;
        break;
    case Family::Qwen36:
    case Family::Qwen38:
        if (!cfg.hidden)
            cfg.hidden = full ? 2048 : 64;
        if (!cfg.n_layers)
            cfg.n_layers = full ? 40 : 2;
        if (!cfg.vocab)
            cfg.vocab = full ? 151936 : 128;
        if (!cfg.n_q_heads)
            cfg.n_q_heads = full ? 16 : 4;
        if (!cfg.n_kv_heads)
            cfg.n_kv_heads = full ? 4 : 2;
        if (!cfg.head_dim)
            cfg.head_dim = 64;
        if (!cfg.dense_intermediate)
            cfg.dense_intermediate = full ? 5632 : 128;
        if (!cfg.moe.n_experts && full)
            cfg.moe.n_experts = 128;
        if (!cfg.moe.topk)
            cfg.moe.topk = 8;
        if (cfg.family == Family::Qwen36 && cfg.n_layers > 0 &&
            static_cast<int>(cfg.is_kda.size()) == cfg.n_layers) {
            bool any = false;
            for (int8_t v : cfg.is_kda)
                if (v)
                    any = true;
            if (!any) {
                cfg.is_kda.assign(cfg.n_layers, 1);
                cfg.is_full.assign(cfg.n_layers, 0);
                for (int i = 3; i < cfg.n_layers; i += 4) {
                    cfg.is_kda[i] = 0;
                    cfg.is_full[i] = 1;
                }
            }
        }
        break;
    case Family::Olmoe:
        if (!cfg.hidden)
            cfg.hidden = full ? 2048 : 64;
        if (!cfg.n_layers)
            cfg.n_layers = full ? 16 : 2;
        if (!cfg.vocab)
            cfg.vocab = full ? 50304 : 128;
        if (!cfg.n_q_heads)
            cfg.n_q_heads = full ? 16 : 4;
        if (!cfg.n_kv_heads)
            cfg.n_kv_heads = full ? 16 : 2;
        if (!cfg.head_dim)
            cfg.head_dim = 64;
        if (!cfg.moe.n_experts && full)
            cfg.moe.n_experts = 64;
        if (!cfg.moe.topk)
            cfg.moe.topk = 8;
        break;
    case Family::Inkling:
        if (!cfg.hidden)
            cfg.hidden = full ? 4096 : 64;
        if (!cfg.n_layers)
            cfg.n_layers = full ? 48 : 2;
        if (!cfg.vocab)
            cfg.vocab = full ? 128000 : 128;
        if (!cfg.n_q_heads)
            cfg.n_q_heads = full ? 32 : 4;
        if (!cfg.n_kv_heads)
            cfg.n_kv_heads = full ? 8 : 2;
        if (!cfg.head_dim)
            cfg.head_dim = 64;
        if (!cfg.sliding_window)
            cfg.sliding_window = 512;
        if (!cfg.moe.n_experts && full)
            cfg.moe.n_experts = 128;
        if (!cfg.moe.topk)
            cfg.moe.topk = 8;
        break;
    default:
        break;
    }
    cfg.mla.rope_theta = cfg.rope_theta;
}

Family sniff_family(const std::string &model_dir, std::string *model_type) {
    std::string raw = read_file(model_dir + "/config.json");
    if (raw.empty()) {
        // MiniMax-H3 snapshots often advertise themselves via layout files.
        if (!read_file(model_dir + "/config.json").size()) {
            std::ifstream dit(model_dir + "/transformer/config.json");
            if (dit)
                raw = "{}";
        }
    }
    std::string blob = raw;
    // Also peek at a few sibling markers.
    if (blob.empty()) {
        std::ifstream marker(model_dir + "/configuration.json");
        if (marker)
            blob = std::string((std::istreambuf_iterator<char>(marker)), {});
    }

    auto lower_has = [&](const char *s) {
        std::string t = blob;
        for (char &c : t)
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char>(c - 'A' + 'a');
        return t.find(s) != std::string::npos;
    };

    json root = json::object();
    if (!raw.empty()) {
        try {
            root = json::parse(raw);
        } catch (...) {
            root = json::object();
        }
    }
    std::string mt = jstr(root, "model_type");
    std::string arch;
    if (root.contains("architectures") && root["architectures"].is_array() &&
        !root["architectures"].empty() && root["architectures"][0].is_string())
        arch = root["architectures"][0].get<std::string>();
    if (model_type)
        *model_type = mt.empty() ? arch : mt;

    auto has = [&](const std::string &s, const char *p) {
        return s.find(p) != std::string::npos;
    };

    if (has(mt, "kimi") || has(arch, "Kimi") || has(mt, "kimi_linear") ||
        has(arch, "KimiLinear"))
        return Family::KimiK3;

    std::string mt_l = mt, arch_l = arch;
    for (char &c : mt_l)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    for (char &c : arch_l)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    // Official colibri types (qwen3_5_moe / qwen4_exp) ship text_config +
    // layer_types. Match them before the GLM catch-all.
    if (has(mt_l, "qwen4_exp") || has(arch_l, "qwen4exp") || has(mt_l, "qwen3.8") ||
        has(mt_l, "qwen38") || has(mt_l, "qwen3_8") || has(arch_l, "qwen3.8") ||
        has(arch_l, "qwen38"))
        return Family::Qwen38;
    if (has(mt_l, "qwen3_5_moe") || has(arch_l, "qwen3_5moe") || has(mt_l, "qwen3.6") ||
        has(mt_l, "qwen36") || has(mt_l, "qwen3_6") || has(arch_l, "qwen3.6") ||
        has(arch_l, "qwen36"))
        return Family::Qwen36;

    if (has(mt, "glm") || has(arch, "Glm") || has(arch, "GLM") ||
        root.contains("text_config")) {
        // GLM-5.3 is the K3-shaped hybrid; GLM-5.2 is also glm* but we treat
        // hybrid/kda markers as 5.3.
        if (root.contains("linear_attn_config") || text_object(root).contains("linear_attn_config") ||
            text_object(root).contains("layer_types") || has(mt, "5.3") || has(arch, "5.3") ||
            has(mt, "glm4") || has(arch, "Glm4"))
            return Family::Glm53;
        if (has(mt, "glm") || has(arch, "Glm"))
            return Family::Glm53;
    }
    if (has(mt, "olmoe") || has(arch, "Olmoe") || has(arch, "OLMoE"))
        return Family::Olmoe;
    if (has(mt, "inkling") || has(arch, "Inkling"))
        return Family::Inkling;
    if (has(mt, "llama") || has(arch, "Llama"))
        return Family::Llama;
    if (has(mt, "minimax") || has(arch, "MiniMax") || has(mt, "h3") || lower_has("dit") ||
        lower_has("minimax-h3") || lower_has("transformer_blocks"))
        return Family::H3;
    {
        std::string mt_l = mt, arch_l = arch;
        for (char &c : mt_l)
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char>(c - 'A' + 'a');
        for (char &c : arch_l)
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char>(c - 'A' + 'a');
        if (has(mt_l, "deepseek") || has(mt_l, "dsv4") || has(arch_l, "deepseek") ||
            has(arch_l, "dsv4") || has(arch, "DeepseekV4") || has(arch, "DeepSeekV4") ||
            has(arch, "DeepSeek-V4"))
            return Family::Dsv4;
    }

    // Directory name fallback.
    std::string dir = model_dir;
    for (char &c : dir)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    if (dir.find("kimi") != std::string::npos || dir.find("k3") != std::string::npos)
        return Family::KimiK3;
    if (dir.find("glm") != std::string::npos)
        return Family::Glm53;
    if (dir.find("minimax") != std::string::npos || dir.find("h3") != std::string::npos)
        return Family::H3;
    if (dir.find("qwen4_exp") != std::string::npos || dir.find("qwen38") != std::string::npos ||
        dir.find("qwen3.8") != std::string::npos)
        return Family::Qwen38;
    if (dir.find("qwen3_5_moe") != std::string::npos || dir.find("qwen36") != std::string::npos ||
        dir.find("qwen3.6") != std::string::npos || dir.find("qwen3.5") != std::string::npos)
        return Family::Qwen36;
    if (dir.find("olmoe") != std::string::npos)
        return Family::Olmoe;
    if (dir.find("inkling") != std::string::npos)
        return Family::Inkling;
    if (dir.find("llama") != std::string::npos)
        return Family::Llama;
    if (dir.find("dsv4") != std::string::npos || dir.find("deepseek-v4") != std::string::npos ||
        dir.find("deepseek_v4") != std::string::npos || dir.find("deepseekv4") != std::string::npos)
        return Family::Dsv4;
    return Family::Unknown;
}

Status load_model_config(const std::string &model_dir, ModelConfig &out, std::string &err) {
    out = ModelConfig{};
    std::string mt;
    out.family = sniff_family(model_dir, &mt);
    out.model_type = mt;

    std::string raw = read_file(model_dir + "/config.json");
    if (raw.empty() && out.family == Family::H3) {
        apply_family_defaults(out);
        out.architecture = "MiniMaxH3";
        return Status::Ok;
    }
    if (raw.empty()) {
        err = "missing config.json in " + model_dir;
        return Status::NotFound;
    }

    json root;
    try {
        root = json::parse(raw);
    } catch (const std::exception &e) {
        err = std::string("config.json parse: ") + e.what();
        return Status::ParseError;
    }

    const json &text = text_object(root);
    out.architecture = jstr(root, "model_type", mt);
    if (root.contains("architectures") && root["architectures"].is_array() &&
        !root["architectures"].empty() && root["architectures"][0].is_string())
        out.architecture = root["architectures"][0].get<std::string>();

    out.hidden = jnum(text, "hidden_size", 0);
    out.n_layers = jnum(text, "num_hidden_layers", 0);
    out.vocab = jnum(text, "vocab_size", 0);
    if (text.contains("first_k_dense_replace")) {
        out.first_dense = jnum(text, "first_k_dense_replace", 0);
    } else if (text.contains("mlp_layer_types") && text["mlp_layer_types"].is_array()) {
        out.first_dense = static_cast<int>(text["mlp_layer_types"].size());
        int i = 0;
        for (const auto &kind : text["mlp_layer_types"]) {
            if (kind.is_string()) {
                const std::string s = kind.get<std::string>();
                if (s.find("sparse") != std::string::npos || s.find("moe") != std::string::npos) {
                    out.first_dense = i;
                    break;
                }
            }
            ++i;
        }
    } else {
        out.first_dense = 0;
    }
    out.dense_intermediate = jnum(text, "intermediate_size", 0);
    out.max_position = jnum(text, "max_position_embeddings", 0);
    if (!out.max_position)
        out.max_position = jnum(text, "original_max_position_embeddings", 0);
    out.rms_eps = jfloat(text, "rms_norm_eps", 1e-5f);
    out.rope_theta = jfloat(text, "rope_theta", 10000.f);
    if (text.contains("rope_parameters") && text["rope_parameters"].is_object())
        out.rope_theta = jfloat(text["rope_parameters"], "rope_theta", out.rope_theta);
    out.bos = jnum(text, "bos_token_id", 0);

    auto ingest_eos = [&](const json &j) {
        if (!j.contains("eos_token_id") || j["eos_token_id"].is_null())
            return;
        const json &e = j["eos_token_id"];
        auto push = [&](int id) {
            if (id < 0)
                return;
            for (int have : out.eos_ids)
                if (have == id)
                    return;
            out.eos_ids.push_back(id);
        };
        if (e.is_number_integer() || e.is_number()) {
            int id = e.is_number_integer() ? e.get<int>() : static_cast<int>(e.get<double>());
            push(id);
        } else if (e.is_array()) {
            for (const auto &el : e) {
                if (el.is_number_integer())
                    push(el.get<int>());
                else if (el.is_number())
                    push(static_cast<int>(el.get<double>()));
            }
        }
        if (!out.eos_ids.empty())
            out.eos = out.eos_ids[0];
    };
    ingest_eos(text);
    ingest_eos(root);
    {
        json gen;
        std::string graw = read_file(model_dir + "/generation_config.json");
        if (!graw.empty()) {
            try {
                gen = json::parse(graw);
                ingest_eos(gen);
            } catch (...) {
            }
        }
    }

    out.mla.n_heads = jnum(text, "num_attention_heads", 0);
    out.mla.q_lora = jnum(text, "q_lora_rank", 0);
    out.mla.kv_lora = jnum(text, "kv_lora_rank", jnum(text, "kv_lora", 0));
    out.mla.qk_nope = jnum(text, "qk_nope_head_dim", 0);
    out.mla.qk_rope = jnum(text, "qk_rope_head_dim", 0);
    out.mla.v_head = jnum(text, "v_head_dim", 0);
    out.n_q_heads = jnum(text, "num_attention_heads", 0);
    out.n_kv_heads = jnum(text, "num_key_value_heads", out.n_q_heads);
    out.head_dim = jnum(text, "head_dim", 0);
    if (!out.head_dim)
        out.head_dim = out.hidden && out.n_q_heads ? out.hidden / out.n_q_heads : 0;
    out.o_lora = jnum(text, "o_lora_rank", 0);
    out.o_groups = jnum(text, "o_groups", 0);
    out.sliding_window = jnum(text, "sliding_window", 0);

    out.moe.n_experts = jnum(text, "num_experts", jnum(text, "n_routed_experts", 0));
    out.moe.topk = jnum(text, "num_experts_per_token", jnum(text, "num_experts_per_tok", 0));
    out.moe.intermediate = jnum(text, "moe_intermediate_size", 0);
    out.moe.latent = jnum(text, "routed_expert_hidden_size", 0);
    out.moe.n_shared = jnum(text, "num_shared_experts", jnum(text, "n_shared_experts", 0));
    out.moe.routed_scale = jfloat(text, "routed_scaling_factor", 1.f);
    out.moe.situ_b1 = jfloat(text, "activation_situ_beta", 4.f);
    out.moe.situ_b2 = jfloat(text, "activation_situ_linear_beta", 25.f);
    out.moe.swiglu_limit = jfloat(text, "swiglu_limit", 0.f);

    out.attn_res.block_size = jnum(text, "attn_res_block_size", 0);
    out.dsa.topk = jnum(text, "index_topk", 0);
    out.dsa.n_heads = jnum(text, "index_n_heads", 0);
    out.dsa.head_dim = jnum(text, "index_head_dim", 0);
    out.dsa.kpool = jnum(text, "index_kpool", out.dsa.kpool);
    if (text.contains("index_kpool_always_select_tail")) {
        if (text["index_kpool_always_select_tail"].is_boolean())
            out.dsa.always_select_tail = text["index_kpool_always_select_tail"].get<bool>();
        else
            out.dsa.always_select_tail = jnum(text, "index_kpool_always_select_tail", 1) != 0;
    }

    if (text.contains("linear_attn_config") && text["linear_attn_config"].is_object()) {
        const json &la = text["linear_attn_config"];
        out.kda.heads = jnum(la, "num_heads", jnum(text, "linear_num_heads", 0));
        out.kda.head_dim = jnum(la, "head_dim", jnum(text, "linear_head_dim", 0));
        out.kda.conv_k = jnum(la, "short_conv_kernel_size",
                              jnum(text, "linear_conv_kernel_dim", 4));
        out.kda.gate_lower_bound = jfloat(la, "gate_lower_bound", -5.f);
        out.kda.full_rank_gate = true;
        if (la.contains("use_full_rank_gate") && la["use_full_rank_gate"].is_boolean())
            out.kda.full_rank_gate = la["use_full_rank_gate"].get<bool>();
    } else {
        out.kda.heads = jnum(text, "linear_num_heads", out.kda.heads);
        out.kda.head_dim = jnum(text, "linear_head_dim", out.kda.head_dim);
        out.kda.conv_k = jnum(text, "linear_conv_kernel_dim", out.kda.conv_k ? out.kda.conv_k : 4);
    }

    out.mhc.mult = jnum(text, "hc_mult", 0);
    out.mhc.iters = jnum(text, "hc_iters", jnum(text, "hc_sinkhorn_iters", 20));
    out.mhc.eps = jfloat(text, "hc_eps", 1e-6f);
    out.dsa.topk = jnum(text, "index_topk", out.dsa.topk);
    out.dsa.n_heads = jnum(text, "index_n_heads", out.dsa.n_heads);
    out.dsa.head_dim = jnum(text, "index_head_dim", out.dsa.head_dim);
    out.dsa.kpool = jnum(text, "index_kpool", out.dsa.kpool);

    if (root.contains("vision_config") && root["vision_config"].is_object()) {
        const json &v = root["vision_config"];
        out.vision.layers = jnum(v, "depth", jnum(v, "num_hidden_layers", 0));
        out.vision.hidden = jnum(v, "hidden_size", 0);
        out.vision.heads = jnum(v, "num_heads", jnum(v, "num_attention_heads", 0));
        out.vision.intermediate = jnum(v, "intermediate_size", 0);
        out.vision.patch = jnum(v, "patch_size", 14);
        out.vision.image_size = jnum(v, "image_size", 448);
        out.vision.merge = jnum(v, "spatial_merge_size", 2);
        out.vision.out_hidden = jnum(v, "out_hidden_size", out.hidden);
        out.vision.temporal = jnum(v, "temporal_patch_size", 2);
        out.vision.in_channels = jnum(v, "in_channels", 3);
        out.vision.proj_intermediate = jnum(v, "projection_intermediate_size", 0);
        out.vision.eps = jfloat(v, "rms_norm_eps", 1e-5f);
        out.vision.swiglu_limit = jfloat(v, "swiglu_limit", 10.f);
        out.vision.rope_theta = jfloat(v, "rope_theta", 10000.f);
    }
    out.vision.image_token = jnum(root, "image_token_id", -1);
    out.vision.image_start_token = jnum(root, "image_start_token_id", -1);
    out.vision.image_end_token = jnum(root, "image_end_token_id", -1);

    // Layer kinds.
    out.is_kda.assign(out.n_layers > 0 ? out.n_layers : 0, 0);
    out.is_full.assign(out.n_layers > 0 ? out.n_layers : 0, 0);
    if (text.contains("layer_types") && text["layer_types"].is_array()) {
        const json &types = text["layer_types"];
        for (int i = 0; i < out.n_layers && i < static_cast<int>(types.size()); ++i) {
            if (!types[i].is_string())
                continue;
            std::string s = types[i].get<std::string>();
            if (s.find("linear") != std::string::npos) {
                out.is_kda[i] = 1;
                out.is_full[i] = 0;
            } else {
                out.is_kda[i] = 0;
                out.is_full[i] = 1;
            }
        }
    } else if (text.contains("linear_attn_config") &&
               text["linear_attn_config"].contains("kda_layers") &&
               text["linear_attn_config"]["kda_layers"].is_array()) {
        out.is_full.assign(out.n_layers, 1);
        out.is_kda.assign(out.n_layers, 0);
        for (const auto &v : text["linear_attn_config"]["kda_layers"]) {
            int idx = v.is_number() ? v.get<int>() : -1;
            // HF lists are often 1-based in K3 tiny config.
            if (idx >= 1 && idx <= out.n_layers) {
                out.is_kda[idx - 1] = 1;
                out.is_full[idx - 1] = 0;
            } else if (idx >= 0 && idx < out.n_layers) {
                out.is_kda[idx] = 1;
                out.is_full[idx] = 0;
            }
        }
    } else if (out.family == Family::KimiK3 && out.n_layers > 0) {
        // Default K3: every 4th + last are MLA; the rest are KDA.
        out.is_kda.assign(out.n_layers, 1);
        out.is_full.assign(out.n_layers, 0);
        for (int i = 3; i < out.n_layers; i += 4) {
            out.is_kda[i] = 0;
            out.is_full[i] = 1;
        }
        out.is_kda[out.n_layers - 1] = 0;
        out.is_full[out.n_layers - 1] = 1;
    } else if (out.family == Family::Glm53 && out.n_layers > 0) {
        // (KDA KDA KDA FULL) repeating.
        out.is_kda.assign(out.n_layers, 1);
        out.is_full.assign(out.n_layers, 0);
        for (int i = 3; i < out.n_layers; i += 4) {
            out.is_kda[i] = 0;
            out.is_full[i] = 1;
        }
    } else if (out.family == Family::Dsv4 && out.n_layers > 0) {
        out.is_kda.assign(out.n_layers, 0);
        out.is_full.assign(out.n_layers, 1);
    }

    if (out.family == Family::Unknown)
        out.family = sniff_family(model_dir, nullptr);

    // Official HF / coli names for DSV4 when the generic keys left a field empty.
    if (out.family == Family::Dsv4) {
        if (!out.moe.n_experts)
            out.moe.n_experts = jnum(text, "n_routed_experts", 0);
        if (!out.moe.topk)
            out.moe.topk = jnum(text, "num_experts_per_tok", 0);
        if (!out.moe.n_shared)
            out.moe.n_shared = jnum(text, "n_shared_experts", 0);
        if (!out.mla.q_lora)
            out.mla.q_lora = jnum(text, "q_lora_rank", 0);
        if (!out.mla.qk_rope)
            out.mla.qk_rope = jnum(text, "qk_rope_head_dim", 0);
        if (!out.mla.kv_lora)
            out.mla.kv_lora = jnum(text, "kv_lora_rank", jnum(text, "kv_lora", 0));
        if (!out.dsa.topk)
            out.dsa.topk = jnum(text, "index_topk", 0);
        if (!out.dsa.n_heads)
            out.dsa.n_heads = jnum(text, "index_n_heads", 0);
        if (!out.dsa.head_dim)
            out.dsa.head_dim = jnum(text, "index_head_dim", 0);
        if (!out.sliding_window)
            out.sliding_window = jnum(text, "sliding_window", 0);
        if (!out.o_lora)
            out.o_lora = jnum(text, "o_lora_rank", 0);
        if (!out.o_groups)
            out.o_groups = jnum(text, "o_groups", 0);
        if (!out.max_position)
            out.max_position = jnum(text, "original_max_position_embeddings",
                                    jnum(text, "max_position_embeddings", 0));
        if (!out.moe.intermediate)
            out.moe.intermediate = jnum(text, "moe_intermediate_size", 0);
        if (out.moe.routed_scale == 1.f)
            out.moe.routed_scale = jfloat(text, "routed_scaling_factor", 1.f);
        if (!out.moe.swiglu_limit)
            out.moe.swiglu_limit = jfloat(text, "swiglu_limit", 0.f);
        if (!out.mhc.mult)
            out.mhc.mult = jnum(text, "hc_mult", 0);
        out.mhc.iters = jnum(text, "hc_sinkhorn_iters", out.mhc.iters);
        out.mhc.eps = jfloat(text, "hc_eps", out.mhc.eps);
        if (out.rms_eps == 1e-5f)
            out.rms_eps = jfloat(text, "rms_norm_eps", out.rms_eps);
        if (out.rope_theta == 10000.f)
            out.rope_theta = jfloat(text, "rope_theta", out.rope_theta);
    }

    apply_family_defaults(out);
    if (out.family == Family::Unknown) {
        err = "could not detect model family in " + model_dir;
        return Status::Unsupported;
    }
    return Status::Ok;
}

RuntimeConfig runtime_from_env() {
    RuntimeConfig rt;
    auto get = [](const char *k) -> const char * {
        return std::getenv(k);
    };
    if (const char *v = get("MVLLM_DEVICE"))
        rt.device = parse_device(v);
    if (const char *v = get("MVLLM_BITS"))
        rt.dense_bits = std::atoi(v);
    if (const char *v = get("MVLLM_HEAD_BITS"))
        rt.head_bits = std::atoi(v);
    if (const char *v = get("MVLLM_MLA_BITS"))
        rt.mla_bits = std::atoi(v);
    if (const char *v = get("MVLLM_EXPERT_GB"))
        rt.expert_gb = std::atof(v);
    if (const char *v = get("MVLLM_LOAD_THREADS"))
        rt.loader_threads = std::atoi(v);
    if (const char *v = get("MVLLM_DIRECT"))
        rt.o_direct = std::atoi(v) != 0;
    if (const char *v = get("MVLLM_PIPE"))
        rt.pipe = std::atoi(v) != 0;
    if (const char *v = get("MVLLM_PORT"))
        rt.host_port = std::atoi(v);
    if (const char *v = get("K3_EXPERT_GB"))
        rt.expert_gb = std::atof(v);
    if (const char *v = get("GLM53_EXPERT_GB"))
        rt.expert_gb = std::atof(v);
    if (const char *v = get("K3_BITS"))
        rt.dense_bits = std::atoi(v);
    if (const char *v = get("K3_MLA_BITS"))
        rt.mla_bits = std::atoi(v);
    if (const char *v = get("GLM53_MLA_BITS"))
        rt.mla_bits = std::atoi(v);
    if (const char *v = get("GLM53_BITS"))
        rt.dense_bits = std::atoi(v);
    if (const char *v = get("DSV4_EXPERT_GB"))
        rt.expert_gb = std::atof(v);
    if (const char *v = get("DSV4_BITS"))
        rt.dense_bits = std::atoi(v);
    if (const char *v = get("DSV4_MLA_BITS"))
        rt.mla_bits = std::atoi(v);
    if (const char *v = get("MVLLM_KV_SLOTS") ? get("MVLLM_KV_SLOTS")
                                              : (get("KV_SLOTS") ? get("KV_SLOTS")
                                                                 : get("COLI_KV_SLOTS"))) {
        rt.kv_slots = std::atoi(v);
        if (rt.kv_slots < 1)
            rt.kv_slots = 1;
        if (rt.kv_slots > 16)
            rt.kv_slots = 16;
    }
    if (const char *v = get("MVLLM_KV") ? get("MVLLM_KV")
                                        : (get("MVLLM_KV_PATH") ? get("MVLLM_KV_PATH")
                                                                : get("COLI_KV")))
        rt.kv_path = v;
    if (const char *v = get("MVLLM_KV_VER") ? get("MVLLM_KV_VER") : get("COLI_KV_VER")) {
        rt.kv_persist_ver = std::atoi(v);
        if (rt.kv_persist_ver < 1 || rt.kv_persist_ver > 3)
            rt.kv_persist_ver = 1;
    }
    if (const char *v = get("COLI_KV8")) {
        if (std::atoi(v) != 0)
            rt.kv_persist_ver = 2;
    }
    if (const char *v = get("COLI_TQ")) {
        if (std::atoi(v) != 0)
            rt.kv_persist_ver = 3;
    }
    if (const char *v = get("COLI_TQ_CODEC") ? get("COLI_TQ_CODEC") : get("MVLLM_KV_TQ_CODEC"))
        rt.kv_tq_codec = std::atoi(v) != 0 ? 1 : 0;
    if (const char *v = get("COLI_TQ_BITS") ? get("COLI_TQ_BITS") : get("MVLLM_KV_TQ_BITS")) {
        rt.kv_tq_bits = std::atoi(v);
        if (rt.kv_tq_bits != 3)
            rt.kv_tq_bits = 4;
    }
    if (const char *v = get("COLI_MAX_QUEUE") ? get("COLI_MAX_QUEUE") : get("MVLLM_MAX_QUEUE")) {
        int n = std::atoi(v);
        if (n >= 0) {
            if (n > 4096)
                n = 4096;
            rt.max_queue = n;
        }
    }
    if (const char *v = get("COLI_QUEUE_TIMEOUT") ? get("COLI_QUEUE_TIMEOUT")
                                                  : get("MVLLM_QUEUE_TIMEOUT")) {
        int n = std::atoi(v);
        if (n > 0) {
            if (n > 86400)
                n = 86400;
            rt.queue_timeout_s = n;
        }
    }
    if (const char *v = get("COLI_KA_GAP") ? get("COLI_KA_GAP") : get("MVLLM_KA_GAP")) {
        double n = std::atof(v);
        if (n > 0) {
            if (n > 3600)
                n = 3600;
            rt.ka_gap_s = n;
        }
    }
    if (const char *v = get("COLI_VISIBLE_KEEPALIVE") ? get("COLI_VISIBLE_KEEPALIVE")
                                                      : get("MVLLM_VISIBLE_KEEPALIVE"))
        rt.visible_keepalive = std::atoi(v) != 0;
    if (const char *v = get("COLI_WEB_DIST") ? get("COLI_WEB_DIST") : get("MVLLM_WEB_DIST")) {
        if (v[0])
            rt.web_dist = v;
    }
    if (const char *v = get("COLI_ALLOWED_HOSTS") ? get("COLI_ALLOWED_HOSTS")
                                                  : get("MVLLM_ALLOWED_HOSTS")) {
        if (v[0])
            rt.allowed_hosts = v;
    }
    return rt;
}

} // namespace mvllm
