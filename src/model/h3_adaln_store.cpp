#include "h3_adaln_store.hpp"
#include "h3_adaln.hpp"
#include "../io/file_io.hpp"

#include <cstdlib>
#include <cstring>

namespace mvllm {

bool h3_adaln_read_w(const io::StHit &w, int rows, int cols, std::vector<float> &dst) {
    if (!w.file || !w.tensor || rows < 1 || cols < 1)
        return false;
    const io::StTensor &t = *w.tensor;
    if (t.shape.size() != 2 || t.shape[1] != cols || t.shape[0] < rows)
        return false;
    const bool is_bf16 = t.dtype == "BF16";
    const bool is_f32 = t.dtype == "F32" || t.dtype == "F32_";
    if (!is_bf16 && !is_f32)
        return false;
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t elem = is_bf16 ? 2 : 4;
    std::vector<uint8_t> raw(n * elem);
    std::string err;
    const int64_t off = io::st_file_offset(*w.file, t);
    if (io::pread_full(w.file->fd, raw.data(), raw.size(), off, err) != Status::Ok)
        return false;
    dst.assign(n, 0.f);
    if (is_f32) {
        std::memcpy(dst.data(), raw.data(), raw.size());
        return true;
    }
    const uint16_t *src = reinterpret_cast<const uint16_t *>(raw.data());
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        dst[i] = v;
    }
    return true;
}

void H3AdalnStore::close() {
    wait_prefetch();
    io::st_close_dir(files_);
    hits_.clear();
    w_res_.clear();
    b_res_.clear();
    w_rows_.clear();
    pref_w_.clear();
    pref_layer_ = -1;
    mode_ = H3AdalnMode::Off;
    three_mod_n_ = 0;
}

bool H3AdalnStore::has(int layer) const {
    if (layer < 0)
        return false;
    if (files_.empty())
        return layer < static_cast<int>(w_res_.size()) && !w_res_[static_cast<size_t>(layer)].empty();
    return layer < static_cast<int>(hits_.size()) && hits_[static_cast<size_t>(layer)].usable;
}

const char *H3AdalnStore::tag() const {
    switch (mode_) {
    case H3AdalnMode::Skip:
        tag_ = "skip";
        break;
    case H3AdalnMode::Stream:
        tag_ = "stream";
        break;
    case H3AdalnMode::Resident:
        tag_ = "resident";
        break;
    case H3AdalnMode::Capped:
        tag_ = "capped-" + std::to_string(cap_);
        break;
    default:
        tag_ = "off";
        break;
    }
    if (skip_n_ > 0 && mode_ != H3AdalnMode::Off && mode_ != H3AdalnMode::Skip)
        tag_ += ",skip=" + std::to_string(skip_n_);
    if (three_mod_n_ > 0 && mode_ != H3AdalnMode::Off && mode_ != H3AdalnMode::Skip)
        tag_ += ",3mod";
    return tag_.c_str();
}

int H3AdalnStore::w_rows(int layer) const {
    if (layer < 0)
        return 0;
    if (layer < static_cast<int>(w_rows_.size()) && w_rows_[static_cast<size_t>(layer)] > 0)
        return w_rows_[static_cast<size_t>(layer)];
    if (!files_.empty() && layer < static_cast<int>(hits_.size()) &&
        hits_[static_cast<size_t>(layer)].usable)
        return hits_[static_cast<size_t>(layer)].w_rows;
    return 0;
}

void H3AdalnStore::wait_prefetch() {
    if (th_.joinable())
        th_.join();
}

void H3AdalnStore::bind(std::vector<io::StFile> &files, int n_blocks, int hidden) {
    close();
    hidden_ = hidden;
    cap_ = 0;
    skip_n_ = 0;
    three_mod_n_ = 0;

    bool force_resident = false;
    if (const char *e = std::getenv("MVLLM_H3_ADALN_RESIDENT"))
        force_resident = std::strcmp(e, "0") != 0;
    const char *max_e = std::getenv("MVLLM_H3_ADALN_MAX");
    const bool max_set = max_e != nullptr;
    const int max_res = max_set ? std::atoi(max_e) : -1;
    if (force_resident)
        mode_ = H3AdalnMode::Resident;
    else if (max_set && max_res == 0)
        mode_ = H3AdalnMode::Skip;
    else if (max_res > 0) {
        mode_ = H3AdalnMode::Capped;
        cap_ = n_blocks > 0 && max_res > n_blocks ? n_blocks : max_res;
        if (n_blocks < 1)
            cap_ = 0;
    } else {
        mode_ = H3AdalnMode::Stream;
    }

    files_ = std::move(files);
    if (n_blocks < 1) {
        if (mode_ != H3AdalnMode::Skip)
            mode_ = H3AdalnMode::Off;
        io::st_close_dir(files_);
        return;
    }

    hits_.assign(static_cast<size_t>(n_blocks), {});
    w_res_.assign(static_cast<size_t>(n_blocks), {});
    b_res_.assign(static_cast<size_t>(n_blocks), {});
    w_rows_.assign(static_cast<size_t>(n_blocks), 0);

    const int need = hidden >= 1 ? kH3AdalnSlots * hidden : 0;
    const int need3 = hidden >= 1 ? h3_adaln_out(hidden) : 0;
    int usable_n = 0;
    for (int i = 0; i < n_blocks; ++i) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        hits_[static_cast<size_t>(i)].w = io::st_find_dir(files_, p + "adaln_proj.linear.weight");
        hits_[static_cast<size_t>(i)].b = io::st_find_dir(files_, p + "adaln_proj.linear.bias");
        const io::StHit &wh = hits_[static_cast<size_t>(i)].w;
        if (wh.tensor) {
            const auto &sh = wh.tensor->shape;
            const std::string &dt = wh.tensor->dtype;
            const bool dtype_ok = dt == "BF16" || dt == "F32" || dt == "F32_";
            if (need > 0 && sh.size() == 2 && sh[0] >= need && sh[1] > 0 && dtype_ok) {
                hits_[static_cast<size_t>(i)].cols = static_cast<int>(sh[1]);
                const int rows = static_cast<int>(sh[0]);
                hits_[static_cast<size_t>(i)].w_rows = rows;
                hits_[static_cast<size_t>(i)].usable = true;
                w_rows_[static_cast<size_t>(i)] = rows;
                ++usable_n;
                if (need3 > 0 && rows >= need3)
                    ++three_mod_n_;
            } else {
                ++skip_n_;
            }
        }
        if (hits_[static_cast<size_t>(i)].usable && hits_[static_cast<size_t>(i)].b.file &&
            hits_[static_cast<size_t>(i)].b.tensor) {
            int64_t n = 1;
            for (int64_t d : hits_[static_cast<size_t>(i)].b.tensor->shape)
                n *= d;
            if (n > 0) {
                b_res_[static_cast<size_t>(i)].assign(static_cast<size_t>(n), 0.f);
                std::string e;
                if (io::st_read_f32(*hits_[static_cast<size_t>(i)].b.file,
                                    *hits_[static_cast<size_t>(i)].b.tensor,
                                    b_res_[static_cast<size_t>(i)].data(), n, e) != Status::Ok)
                    b_res_[static_cast<size_t>(i)].clear();
            }
        }
        if (hits_[static_cast<size_t>(i)].usable && mode_ != H3AdalnMode::Skip &&
            (mode_ == H3AdalnMode::Resident || i < cap_)) {
            const int prefix = hits_[static_cast<size_t>(i)].w_rows;
            const int take = (need3 > 0 && prefix > need3) ? need3 : prefix;
            h3_adaln_read_w(hits_[static_cast<size_t>(i)].w, take,
                            hits_[static_cast<size_t>(i)].cols, w_res_[static_cast<size_t>(i)]);
        }
    }

    if (mode_ != H3AdalnMode::Skip && usable_n == 0)
        mode_ = H3AdalnMode::Off;

    if (mode_ == H3AdalnMode::Skip) {
        io::st_close_dir(files_);
        hits_.clear();
        w_res_.clear();
        b_res_.clear();
        w_rows_.clear();
        three_mod_n_ = 0;
        return;
    }
    if (mode_ == H3AdalnMode::Resident || mode_ == H3AdalnMode::Off) {
        io::st_close_dir(files_);
        hits_.clear();
    }
}

void H3AdalnStore::prefetch(int layer, int td, int mrows) {
    wait_prefetch();
    pref_layer_ = -1;
    if (layer < 0 || !has(layer))
        return;
    if (layer < static_cast<int>(w_res_.size()) && !w_res_[static_cast<size_t>(layer)].empty())
        return;
    if (files_.empty())
        return;
    if (layer >= static_cast<int>(hits_.size()) || hits_[static_cast<size_t>(layer)].cols != td ||
        mrows < 1)
        return;
    th_ = std::thread([=] {
        std::vector<float> tmp;
        pref_st_ = h3_adaln_read_w(hits_[static_cast<size_t>(layer)].w, mrows, td, tmp)
                       ? Status::Ok
                       : Status::IoError;
        std::lock_guard<std::mutex> g(mu_);
        if (pref_st_ == Status::Ok) {
            pref_w_.swap(tmp);
            pref_layer_ = layer;
            pref_rows_ = mrows;
            pref_cols_ = td;
        }
    });
}

bool H3AdalnStore::load_mod(int layer, const float *temb, int td, int mrows,
                           std::vector<float> &mod, int temb_rows) {
    wait_prefetch();
    std::lock_guard<std::mutex> g(mu_);
    auto fail = [&]() {
        mod.clear();
        return false;
    };
    if (!has(layer) || !temb || td < 1 || mrows < 1 || temb_rows < 1)
        return fail();
    const float *W = nullptr;
    std::vector<float> tmp;
    if (layer < static_cast<int>(w_res_.size()) && !w_res_[static_cast<size_t>(layer)].empty()) {
        if (static_cast<int>(w_res_[static_cast<size_t>(layer)].size()) < mrows * td)
            return fail();
        W = w_res_[static_cast<size_t>(layer)].data();
    } else if (pref_layer_ == layer && pref_rows_ == mrows && pref_cols_ == td &&
               pref_st_ == Status::Ok && pref_w_.size() >= static_cast<size_t>(mrows) * td) {
        tmp.swap(pref_w_);
        pref_layer_ = -1;
        W = tmp.data();
    } else if (!files_.empty() && layer < static_cast<int>(hits_.size()) &&
               hits_[static_cast<size_t>(layer)].usable &&
               hits_[static_cast<size_t>(layer)].cols == td) {
        if (!h3_adaln_read_w(hits_[static_cast<size_t>(layer)].w, mrows, td, tmp))
            return fail();
        W = tmp.data();
    } else {
        return fail();
    }
    if (!W)
        return fail();
    mod.assign(static_cast<size_t>(temb_rows) * static_cast<size_t>(mrows), 0.f);
    const float *bias = (layer < static_cast<int>(b_res_.size()) &&
                         b_res_[static_cast<size_t>(layer)].size() >= static_cast<size_t>(mrows))
                            ? b_res_[static_cast<size_t>(layer)].data()
                            : nullptr;
    if (!h3_adaln_mod(temb, temb_rows, td, W, bias, mrows, mod.data()))
        return fail();
    return true;
}

} // namespace mvllm
