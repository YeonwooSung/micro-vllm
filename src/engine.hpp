#pragma once

#include "core/config.hpp"
#include "model/family.hpp"
#include "serve/scheduler.hpp"
#include "serve/session.hpp"
#include "store/kv_persist.hpp"
#include "store/kv_persist_v2.hpp"
#include "store/kv_persist_v3.hpp"
#include "store/kv_prefix.hpp"
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
    std::string decode_token(int id) const {
        std::string s;
        std::vector<int> one{id};
        tok_.decode(one, s);
        return s;
    }
    void expert_stats(ExpertStoreStats &out) const {
        if (impl_)
            impl_->expert_stats(out);
        else
            out = {};
    }
    void route_telem(RouteTelem &out, bool consume_hits = true) {
        if (impl_)
            impl_->route_telem(out, consume_hits);
        else
            out = {};
        if (consume_hits && out.rows > 0)
            ++hits_seq_;
    }
    int hits_seq() const { return hits_seq_; }
    uint64_t block_hits() const { return impl_ ? impl_->block_hits() : 0; }
    uint64_t block_misses() const { return impl_ ? impl_->block_misses() : 0; }
    SessionStore &sessions() { return sessions_; }
    const SessionStore &sessions() const { return sessions_; }
    BatchScheduler &scheduler() { return sched_; }
    FamilyEngine *family_impl() { return impl_.get(); }

    Status persist_open(const std::string &path, int ver, std::string &err, int slot = 0);
    void persist_close();
    int persist_nrec() const;
    const std::vector<int> &persist_hist() const;
    int prefix_reuse_len(int slot) const; // KvPrefix::len for slot
    // Official all-or-nothing reuse; seeds the slot from session hist if empty.
    int prefix_match(int slot, const std::vector<int> &ids);
    void prefix_commit(int slot, const std::vector<int> &hist);
    Status persist_commit(int slot, const std::vector<int> &hist, std::string &err);

private:
    KvPersistConfig make_persist_cfg() const;
    Status persist_append_tail(int slot, const std::vector<int> &hist, std::string &err);
    Family family_ = Family::Unknown;
    ModelConfig cfg_{};
    RuntimeConfig rt_{};
    Tokenizer tok_;
    std::unique_ptr<FamilyEngine> impl_;
    std::string model_id_;
    std::string model_dir_;
    SessionStore sessions_;
    BatchScheduler sched_;
    KvPersist persist_v1_;
    KvPersistV2 persist_v2_;
    KvPersistV3 persist_v3_;
    KvPrefix prefixes_[kMaxKvSlots];
    std::vector<int> persist_hist_;
    int persist_ver_ = 0;
    int persist_slot_ = 0;
    std::string persist_path_;
    bool persist_open_ = false;
    int hits_seq_ = 0;
};

} // namespace mvllm
