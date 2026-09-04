#include "engine.hpp"
#include "io/shard_probe.hpp"
#include "serve/http_server.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void usage() {
    std::cerr
        << "micro-vllm — consumer-hardware inference server\n"
        << "  micro-vllm info     --model DIR\n"
        << "  micro-vllm smoke    --model DIR\n"
        << "  micro-vllm generate --model DIR --prompt TEXT [--n N] [--chat] [--think|--no-think]\n"
        << "  micro-vllm serve    --model DIR [--host H] [--port P]\n"
        << "  micro-vllm video    --model DIR --prompt TEXT [-o FILE]\n"
        << "\n"
        << "Families: llama | kimi_k3 | glm53 | h3\n"
        << "  --device cpu|metal|cuda   expert GEMM / H3 DiT backend (default cpu)\n"
        << "\n"
        << "Env: MVLLM_EXPERT_GB MVLLM_BITS MVLLM_HEAD_BITS MVLLM_MLA_BITS MVLLM_DEVICE MVLLM_PORT\n"
        << "     K3_EXPERT_GB GLM53_EXPERT_GB K3_BITS GLM53_BITS K3_MLA_BITS\n";
}

std::string arg(int argc, char **argv, const char *key, const char *def = "") {
    for (int i = 0; i < argc - 1; ++i) {
        if (std::strcmp(argv[i], key) == 0)
            return argv[i + 1];
    }
    return def;
}

bool has(int argc, char **argv, const char *key) {
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0)
            return true;
    }
    return false;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || has(argc, argv, "-h") || has(argc, argv, "--help")) {
        usage();
        return argc < 2 ? 1 : 0;
    }
    const std::string cmd = argv[1];
    std::string model = arg(argc, argv, "--model");
    if (model.empty())
        model = arg(argc, argv, "-m");
    if (cmd != "help" && model.empty()) {
        std::cerr << "missing --model DIR\n";
        usage();
        return 1;
    }

    mvllm::RuntimeConfig rt = mvllm::runtime_from_env();
    if (const std::string p = arg(argc, argv, "--port"); !p.empty())
        rt.host_port = std::stoi(p);
    if (const std::string h = arg(argc, argv, "--host"); !h.empty())
        rt.host = h;
    if (const std::string g = arg(argc, argv, "--expert-gb"); !g.empty())
        rt.expert_gb = std::stod(g);
    if (const std::string d = arg(argc, argv, "--device"); !d.empty())
        rt.device = mvllm::parse_device(d);

    if (cmd == "smoke") {
        mvllm::ShardReport rep;
        std::string serr;
        mvllm::Status sst = mvllm::probe_shards(model, rt, rep, serr);
        if (sst != mvllm::Status::Ok) {
            std::cerr << "smoke failed: " << serr << "\n";
            return 1;
        }
        std::cout << mvllm::format_shard_report(rep);
        return rep.shards_ok ? 0 : 1;
    }

    mvllm::Engine engine;
    std::string err;
    mvllm::Status st = engine.load(model, rt, err);
    if (st != mvllm::Status::Ok) {
        std::cerr << "load failed: " << err << " (" << mvllm::status_name(st) << ")\n";
        return 1;
    }

    if (cmd == "info") {
        std::cout << engine.info();
        return 0;
    }

    if (cmd == "generate") {
        std::string prompt = arg(argc, argv, "--prompt");
        if (prompt.empty())
            prompt = arg(argc, argv, "-p", "Hello");
        int n = 16;
        std::string ns = arg(argc, argv, "--n");
        if (ns.empty())
            ns = arg(argc, argv, "-n", "16");
        if (!ns.empty())
            n = std::stoi(ns);
        mvllm::GenParams gp;
        gp.max_new_tokens = n;
        gp.eos = engine.config().eos;
        gp.apply_template = has(argc, argv, "--chat");
        gp.think = has(argc, argv, "--think");
        if (has(argc, argv, "--no-think"))
            gp.think = false;
        else if (gp.apply_template && engine.family() == mvllm::Family::Glm53 &&
                 !has(argc, argv, "--think"))
            gp.think = true;
        mvllm::GenResult out;
        st = engine.generate(prompt, gp, out, err);
        if (st != mvllm::Status::Ok) {
            std::cerr << "generate failed: " << err << "\n";
            return 1;
        }
        std::cout << out.text << "\n";
        std::cerr << "prompt=" << out.prompt_tokens << " completion=" << out.completion_tokens
                  << " ids=";
        for (int t : out.tokens)
            std::cerr << t << " ";
        std::cerr << "\n";
        return 0;
    }

    if (cmd == "video") {
        mvllm::H3GenParams hp;
        hp.prompt = arg(argc, argv, "--prompt");
        if (hp.prompt.empty())
            hp.prompt = arg(argc, argv, "-p", "a red fox");
        hp.output_path = arg(argc, argv, "-o", "h3_out.txt");
        if (const std::string s = arg(argc, argv, "--steps"); !s.empty())
            hp.steps = std::stoi(s);
        if (const std::string s = arg(argc, argv, "--layers"); !s.empty())
            hp.dit_layers = std::stoi(s);
        hp.ssd_streaming = true;
        mvllm::H3GenResult out;
        st = engine.generate_video(hp, out, err);
        if (st != mvllm::Status::Ok) {
            std::cerr << "video failed: " << err << "\n";
            return 1;
        }
        std::cout << out.output_path << "\n" << out.note << "\n"
                  << "blocks=" << out.blocks_streamed << " steps=" << out.steps_run << "\n";
        return 0;
    }

    if (cmd == "serve") {
        mvllm::HttpServer srv;
        st = srv.bind(rt.host, rt.host_port, err);
        if (st != mvllm::Status::Ok) {
            std::cerr << "bind failed: " << err << "\n";
            return 1;
        }
        srv.set_engine(&engine);
        std::cerr << engine.info();
        std::cerr << "listening on http://" << rt.host << ":" << srv.port() << "\n";
        st = srv.serve_forever(err);
        if (st != mvllm::Status::Ok) {
            std::cerr << "serve: " << err << "\n";
            return 1;
        }
        return 0;
    }

    std::cerr << "unknown command: " << cmd << "\n";
    usage();
    return 1;
}
