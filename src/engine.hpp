#pragma once

#include "core/config.hpp"
#include "model/family.hpp"
#include "tok/tokenizer.hpp"

#include <memory>
#include <string>
#include <vector>

namespace mvllm {

class Engine {
public:
    Status load(const std::string &model_dir, const RuntimeConfig &rt, std::string &err);
    Status generate(const std::string &prompt, const GenParams &gp, GenResult &out,
                    std::string &err);
    Status generate_chat(const std::vector<ChatMessage> &msgs, const GenParams &gp, GenResult &out,
                         std::string &err);
    Status generate_ids(const std::vector<int> &ids, const GenParams &gp, GenResult &out,
                        std::string &err);
    Status generate_video(const H3GenParams &hp, H3GenResult &out, std::string &err);
    std::string info() const;
    Family family() const { return family_; }
    const ModelConfig &config() const { return cfg_; }
    const RuntimeConfig &runtime() const { return rt_; }
    const std::string &model_id() const { return model_id_; }
    void expert_stats(ExpertStoreStats &out) const {
        if (impl_)
            impl_->expert_stats(out);
        else
            out = {};
    }
    uint64_t block_hits() const { return impl_ ? impl_->block_hits() : 0; }
    uint64_t block_misses() const { return impl_ ? impl_->block_misses() : 0; }

private:
    Family family_ = Family::Unknown;
    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    Tokenizer tok_;
    std::unique_ptr<FamilyEngine> impl_;
    std::string model_id_;
    std::string model_dir_;
};

} // namespace mvllm
