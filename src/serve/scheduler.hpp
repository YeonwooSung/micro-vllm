#pragma once

#include "../core/types.hpp"
#include "../model/family.hpp"
#include "session.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mvllm {

class Engine;

enum class BatchJobState { Queued, Running, Done, Stopped, Cancelled };

struct BatchJob {
    uint64_t id = 0;
    int slot = 0;
    std::vector<int> ids;
    GenParams gp;
    GenResult out;
    Status status = Status::Ok;
    std::string err;
    BatchJobState state = BatchJobState::Queued;
    int reuse = 0;
};

// Prefill is serial; decode is round-robin across active slots (official mux:
// every active slot contributes one decode step per pump). Falls back to a
// single generate() when the family has no next_token hook.
class BatchScheduler {
public:
    void bind(Engine *engine, SessionStore *sessions);
    void configure(int max_queue, int queue_timeout_s);

    // Enqueue. SLOT_BUSY if slot is held. Returns 0 on failure.
    uint64_t submit(int slot, const std::vector<int> &ids, const GenParams &gp, std::string &err);
    bool stop(uint64_t id);   // finish through DONE
    bool cancel(uint64_t id); // CANCELLED, keep history
    // Run at least one unit of work: one prefill or one decode step on each
    // active job. Returns number of jobs still not terminal.
    int pump();
    bool finished(uint64_t id) const;
    const BatchJob *job(uint64_t id) const;
    int queue_depth() const;
    int running_count() const; // Running + Stopped (slot still held)
    int queued_count() const;  // Queued only
    int max_queue() const;
    int n_jobs() const;        // jobs_.size(), including terminal

private:
    Engine *engine_ = nullptr;
    SessionStore *sessions_ = nullptr;
    int max_queue_ = 8;
    int queue_timeout_s_ = 300;
    uint64_t next_id_ = 1;
    std::vector<BatchJob> jobs_;
};

} // namespace mvllm
