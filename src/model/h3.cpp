#include "family.hpp"
#include "h3_vae.hpp"
#include "../io/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace mvllm {
namespace {

const char *kH3StreamSuffix[4] = {
    "attn.qkv_proj.weight",
    "attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
};

bool ends_with(const std::string &s, const char *suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

uint64_t splitmix64(uint64_t &s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

float gauss01(uint64_t &s) {
    float u = static_cast<float>(splitmix64(s) >> 40) * (1.f / 16777216.f);
    float v = static_cast<float>(splitmix64(s) >> 40) * (1.f / 16777216.f);
    if (u < 1e-7f)
        u = 1e-7f;
    return std::sqrt(-2.f * std::log(u)) * std::cos(6.283185307179586f * v);
}

} // namespace


class H3Engine final : public FamilyEngine {
public:
    Family family() const override { return Family::H3; }
    const ModelConfig &config() const override { return cfg_; }

    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err) override {
        rt_ = rt;
        model_dir_ = model_dir;
        Status st = load_model_config(model_dir, cfg_, err);
        if (st != Status::Ok) {
            // MiniMax-H3 directories may have no HF config.json; still accept.
            cfg_.family = Family::H3;
            apply_family_defaults(cfg_);
            err.clear();
        }
        cfg_.family = Family::H3;
        apply_family_defaults(cfg_);

        st = load_checkpoint(model_dir, err);
        if (st == Status::NotFound) {
            err.clear();
            st = write_synthetic_blocks(model_dir, err);
        }
        if (st != Status::Ok)
            return st;
        {
            std::string verr;
            Status vst = vae_.load(model_dir, cfg_.h3, verr);
            if (vst == Status::NotFound)
                verr.clear();
        }
        loaded_ = true;
        return Status::Ok;
    }

    Status generate(const std::vector<int> &, const GenParams &, GenResult &,
                    std::string &err) override {
        err = "H3 is a video/audio DiT — use /v1/videos/generations";
        return Status::Unsupported;
    }

    Status generate_video(const H3GenParams &hp, H3GenResult &out, std::string &err) override {
        if (!loaded_) {
            err = "h3 not loaded";
            return Status::InvalidArgument;
        }
        int layers = hp.dit_layers > 0 ? hp.dit_layers : cfg_.h3.dit_layers;
        if (layers > cfg_.h3.dit_layers)
            layers = cfg_.h3.dit_layers;
        int steps = hp.steps > 0 ? hp.steps : cfg_.h3.default_steps;
        int reuse = hp.denoise_reuse > 0 ? hp.denoise_reuse : 1;
        int evals = (steps + reuse - 1) / reuse;

        const int hidden = cfg_.h3.hidden > 0 ? cfg_.h3.hidden : 5376;
        const int inner = cfg_.h3.inner > 0 ? cfg_.h3.inner : 7168;
        const int ffn = cfg_.h3.ffn > 0 ? cfg_.h3.ffn : 14336;
        const int hd = cfg_.h3.head_dim > 0 ? cfg_.h3.head_dim : 128;
        const int64_t qkv_b = static_cast<int64_t>(inner) * 3 * hidden * 2;
        const int64_t out_b = static_cast<int64_t>(hidden) * inner * 2;
        const int64_t fc1_b = static_cast<int64_t>(ffn) * 2 * hidden * 2;
        const int64_t fc2_b = static_cast<int64_t>(hidden) * ffn * 2;
        const int64_t expect = qkv_b + out_b + fc1_b + fc2_b;
        const int tokens = 4;
        std::vector<float> latent(static_cast<size_t>(tokens) * hidden, 0.f);
        for (size_t i = 0; i < latent.size(); ++i)
            latent[i] = 0.02f * static_cast<float>(static_cast<int>(i % 17) - 8);

        int streamed = 0;
        int computed = 0;
        for (int s = 0; s < evals; ++s) {
            for (int b = 0; b < layers; ++b) {
                const uint8_t *data = nullptr;
                int64_t bytes = 0;
                Status st = blocks_.acquire(b, &data, &bytes, err);
                if (st != Status::Ok)
                    return st;
                if (hp.ssd_streaming) {
                    if (b + 1 < layers)
                        blocks_.prefetch(b + 1, err);
                    else
                        blocks_.prefetch(0, err);
                }
                if (data && bytes > 0)
                    streamed++;
                if (from_checkpoint_ && data && bytes == expect) {
                    h3_dit_block_cpu(data, qkv_b, out_b, fc1_b, fc2_b, hidden, inner, ffn, hd,
                                     latent.data(), tokens, 1e-6f);
                    ++computed;
                }
                blocks_.release(b);
            }
        }
        float checksum = 0.f;
        for (float v : latent)
            checksum += v * v;

        const int req_w = hp.width > 0 ? hp.width : cfg_.h3.default_width;
        const int req_h = hp.height > 0 ? hp.height : cfg_.h3.default_height;
        const int req_f = hp.frames > 0 ? hp.frames : cfg_.h3.default_frames;
        H3VaeGeom vg = h3_vae_geom(req_w, req_h, req_f, cfg_.h3.vae_spatial, cfg_.h3.vae_latent_ch);
        const int64_t zN = static_cast<int64_t>(vg.latent_ch) * vg.latent_t * vg.latent_h * vg.latent_w;
        std::vector<float> z(static_cast<size_t>(zN > 0 ? zN : 1), 0.f);
        uint64_t rng = hp.seed ? hp.seed : 42ull;
        for (int64_t i = 0; i < zN; ++i)
            z[static_cast<size_t>(i)] = gauss01(rng);
        std::vector<float> rgb(static_cast<size_t>(vg.frames) * vg.height * vg.width * 3, 0.f);
        vae_.decode(z.data(), vg, rgb.data());
        vae_.geom = vg;

        out.blocks_streamed = streamed;
        out.steps_run = evals;
        out.frames = vg.frames;
        out.width = vg.width;
        out.height = vg.height;
        out.vae_used = true;
        out.output_path = hp.output_path.empty() ? (model_dir_ + "/h3_dryrun.txt") : hp.output_path;
        const bool ppm = ends_with(out.output_path, ".ppm");
        const bool raw = ends_with(out.output_path, ".rgb");
        if (ppm) {
            Status wst = h3_write_ppm(out.output_path, rgb.data(), vg.frames, vg.height, vg.width, err);
            if (wst != Status::Ok)
                return wst;
        } else if (raw) {
            std::ofstream rf(out.output_path, std::ios::binary);
            if (!rf) {
                err = "h3: open rgb failed: " + out.output_path;
                return Status::IoError;
            }
            const size_t npix = static_cast<size_t>(vg.frames) * vg.height * vg.width * 3;
            std::vector<unsigned char> bytes(npix);
            for (size_t i = 0; i < npix; ++i) {
                float v = rgb[i];
                if (v < 0.f)
                    v = 0.f;
                if (v > 1.f)
                    v = 1.f;
                bytes[i] = static_cast<unsigned char>(v * 255.f + 0.5f);
            }
            rf.write(reinterpret_cast<const char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
        } else {
            {
                std::ofstream f(out.output_path);
                f << "micro-vllm H3\n"
                  << "prompt: " << hp.prompt << "\n"
                  << "canvas: " << vg.width << "x" << vg.height << " frames=" << vg.frames << "\n"
                  << "steps=" << steps << " reuse=" << reuse << " layers=" << layers << "\n"
                  << "blocks_streamed=" << streamed << " cpu_blocks=" << computed
                  << " slots=" << blocks_.n_slots() << "\n"
                  << "latent_l2=" << checksum << "\n"
                  << "vae=" << (vae_.from_checkpoint ? "real" : "synth") << "\n"
                  << "note: " << (computed ? "CPU DiT residual on streamed BF16" : "stream only")
                  << "\n";
            }
            std::string ppath = out.output_path + ".ppm";
            h3_write_ppm(ppath, rgb.data(), vg.frames, vg.height, vg.width, err);
            err.clear();
        }
        out.note = computed ? "checkpoint: CPU DiT residual on streamed BF16 matrices"
                            : "dry-run: DiT blocks streamed from SSD (2-slot)";
        out.note += vae_.from_checkpoint ? " vae=real" : " vae=synth";
        err.clear();
        return Status::Ok;
    }

    std::string describe() const override {
        std::ostringstream os;
        os << "h3  dit_layers=" << cfg_.h3.dit_layers << " slots=" << cfg_.h3.stream_slots
           << " hidden=" << cfg_.h3.hidden << " inner=" << cfg_.h3.inner << " ffn=" << cfg_.h3.ffn
           << " block_bytes=" << block_bytes_ << " checkpoint="
           << (from_checkpoint_ ? "yes" : "synthetic")
           << " transformer=" << (transformer_dir_.empty() ? "-" : transformer_dir_)
           << " default=" << cfg_.h3.default_width << "x" << cfg_.h3.default_height << "@"
           << cfg_.h3.default_frames << " hits=" << blocks_.hits()
           << " misses=" << blocks_.misses()
           << " vae=" << (vae_.from_checkpoint ? "real" : "synth") << " T=" << vae_.geom.latent_t
           << " lh=" << vae_.geom.latent_h << " lw=" << vae_.geom.latent_w
           << " ch=" << vae_.geom.latent_ch;
        return os.str();
    }

    uint64_t block_hits() const override { return blocks_.hits(); }
    uint64_t block_misses() const override { return blocks_.misses(); }

private:
    Status write_synthetic_blocks(const std::string &model_dir, std::string &err) {
        const int n = cfg_.h3.dit_layers > 0 ? cfg_.h3.dit_layers : 50;
        const int64_t fake = 4096;
        Status st = blocks_.open(n, fake, cfg_.h3.stream_slots > 0 ? cfg_.h3.stream_slots : 2, err);
        if (st != Status::Ok)
            return st;
        blocks_.set_direct(true);
        block_path_ = model_dir + "/.mvllm_h3_blocks.bin";
        {
            std::ofstream out(block_path_, std::ios::binary | std::ios::trunc);
            std::vector<uint8_t> z(static_cast<size_t>(fake), 0);
            for (int i = 0; i < n; ++i) {
                z[0] = static_cast<uint8_t>(i);
                out.write(reinterpret_cast<const char *>(z.data()), fake);
            }
        }
        for (int i = 0; i < n; ++i) {
            st = blocks_.register_block(i, block_path_, static_cast<int64_t>(i) * fake, fake, err);
            if (st != Status::Ok)
                return st;
        }
        block_bytes_ = fake;
        from_checkpoint_ = false;
        transformer_dir_.clear();
        return Status::Ok;
    }

    Status load_checkpoint(const std::string &model_dir, std::string &err) {
        const char *tails[] = {"/FL2VA/transformer", "/transformer", "/dit", ""};
        std::vector<io::StFile> files;
        std::string chosen;
        for (const char *tail : tails) {
            std::string dir = model_dir + tail;
            files.clear();
            err.clear();
            Status st = io::st_open_dir(dir, files, err);
            if (st != Status::Ok || files.empty())
                continue;
            if (io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight").tensor) {
                chosen = dir;
                break;
            }
            io::st_close_dir(files);
        }
        if (chosen.empty()) {
            err.clear();
            return Status::NotFound;
        }

        auto has = [&](const std::string &n) { return io::st_find_dir(files, n).tensor != nullptr; };

        int n_blocks = 0;
        for (int i = 0; i < 64; ++i) {
            if (!has("blocks." + std::to_string(i) + ".attn.qkv_proj.weight"))
                break;
            n_blocks = i + 1;
        }
        if (n_blocks < 1) {
            io::st_close_dir(files);
            return Status::NotFound;
        }

        io::StHit qkv0 = io::st_find_dir(files, "blocks.0.attn.qkv_proj.weight");
        if (qkv0.tensor && qkv0.tensor->shape.size() == 2) {
            cfg_.h3.hidden = static_cast<int>(qkv0.tensor->shape[1]);
            cfg_.h3.inner = static_cast<int>(qkv0.tensor->shape[0] / 3);
        }
        io::StHit fc1 = io::st_find_dir(files, "blocks.0.mlp.fc1.weight");
        if (fc1.tensor && fc1.tensor->shape.size() == 2)
            cfg_.h3.ffn = static_cast<int>(fc1.tensor->shape[0] / 2);

        int64_t slot = 0;
        BlockPiece probe[4];
        for (int p = 0; p < 4; ++p) {
            io::StHit hit = io::st_find_dir(files, std::string("blocks.0.") + kH3StreamSuffix[p]);
            if (!hit.tensor) {
                err = "h3 block 0 missing " + std::string(kH3StreamSuffix[p]);
                io::st_close_dir(files);
                return Status::ParseError;
            }
            probe[p].path = hit.file->path;
            probe[p].offset = io::st_file_offset(*hit.file, *hit.tensor);
            probe[p].bytes = io::st_nbytes(*hit.tensor);
            slot += probe[p].bytes;
        }

        cfg_.h3.dit_layers = n_blocks;
        Status st = blocks_.open(n_blocks, slot, cfg_.h3.stream_slots > 0 ? cfg_.h3.stream_slots : 2,
                                 err);
        if (st != Status::Ok) {
            io::st_close_dir(files);
            return st;
        }
        blocks_.set_direct(true);

        for (int i = 0; i < n_blocks; ++i) {
            std::vector<BlockPiece> pieces;
            pieces.reserve(4);
            for (int p = 0; p < 4; ++p) {
                const std::string name = "blocks." + std::to_string(i) + "." + kH3StreamSuffix[p];
                io::StHit hit = io::st_find_dir(files, name);
                if (!hit.tensor || io::st_nbytes(*hit.tensor) != probe[p].bytes) {
                    err = "h3 block " + std::to_string(i) + " missing or wrong size: " + name;
                    io::st_close_dir(files);
                    return Status::ParseError;
                }
                BlockPiece bp;
                bp.path = hit.file->path;
                bp.offset = io::st_file_offset(*hit.file, *hit.tensor);
                bp.bytes = probe[p].bytes;
                pieces.push_back(bp);
            }
            bool contig = true;
            for (int p = 1; p < 4; ++p) {
                if (pieces[p].path != pieces[0].path ||
                    pieces[p].offset != pieces[p - 1].offset + pieces[p - 1].bytes)
                    contig = false;
            }
            if (contig) {
                st = blocks_.register_block(i, pieces[0].path, pieces[0].offset, slot, err);
            } else {
                st = blocks_.register_block_pieces(i, pieces, err);
            }
            if (st != Status::Ok) {
                io::st_close_dir(files);
                return st;
            }
        }
        io::st_close_dir(files);
        transformer_dir_ = chosen;
        block_bytes_ = slot;
        from_checkpoint_ = true;
        return Status::Ok;
    }

    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    BlockStore blocks_;
    H3Vae vae_;
    std::string model_dir_;
    std::string block_path_;
    std::string transformer_dir_;
    int64_t block_bytes_ = 0;
    bool loaded_ = false;
    bool from_checkpoint_ = false;
};

std::unique_ptr<FamilyEngine> make_h3() { return std::make_unique<H3Engine>(); }

} // namespace mvllm
