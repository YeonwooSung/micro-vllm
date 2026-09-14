#pragma once

#include "../core/types.hpp"
#include "../io/safetensors.hpp"

#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mvllm {

enum class H3AdalnMode { Off, Skip, Stream, Resident, Capped };

struct H3AdalnHit {
    io::StHit w;
    io::StHit b;
    int cols = 0;
    int w_rows = 0;
    bool usable = false;
};

// C-contiguous [O,I] prefix of a rank-2 BF16/F32 tensor. Requires shape[1]==cols.
bool h3_adaln_read_w(const io::StHit &w, int rows, int cols, std::vector<float> &dst);

class H3AdalnStore {
public:
    H3AdalnStore() = default;
    ~H3AdalnStore() { close(); }
    H3AdalnStore(const H3AdalnStore &) = delete;
    H3AdalnStore &operator=(const H3AdalnStore &) = delete;

    void close();
    // Always: files_ = std::move(files); then st_find_dir(files_, name).
    void bind(std::vector<io::StFile> &files, int n_blocks, int hidden);

    bool has(int layer) const;
    H3AdalnMode mode() const { return mode_; }
    int skip_count() const { return skip_n_; }
    // Usable W row count (0 if !has). Survives Resident hits_.clear().
    int w_rows(int layer) const;
    const char *tag() const;

    // Fills mod[temb_rows * mrows] via h3_adaln_mod. Every false path does mod.clear().
    bool load_mod(int layer, const float *temb, int td, int mrows, std::vector<float> &mod,
                  int temb_rows = 1);

    void prefetch(int layer, int td, int mrows);
    void wait_prefetch();

private:
    std::vector<io::StFile> files_;
    std::vector<H3AdalnHit> hits_;
    std::vector<std::vector<float>> w_res_;
    std::vector<std::vector<float>> b_res_;
    std::vector<int> w_rows_;
    H3AdalnMode mode_ = H3AdalnMode::Off;
    int cap_ = 0;
    int hidden_ = 0;
    int skip_n_ = 0;
    int three_mod_n_ = 0;
    mutable std::string tag_;

    std::mutex mu_;
    std::thread th_;
    Status pref_st_ = Status::Ok;
    int pref_layer_ = -1;
    int pref_rows_ = 0;
    int pref_cols_ = 0;
    std::vector<float> pref_w_;
};

} // namespace mvllm
