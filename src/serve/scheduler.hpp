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
    double queued_at = 0; // steady_clock seconds
    double wait_s = 0;    // filled when the job starts Running
};

struct SchedulerSnapshot {
    int active = 0;      // running_count
    int queued = 0;      // queued_count
    int capacity = 1;    // 1 if no sessions else sessions n_slots
    int busy_slots = 0;  // SessionStore::busy_count; 0 if unbound
    int idle = 0;        // max(0, capacity - busy_slots)
    int hist_tokens = 0; // SessionStore::history_total; 0 if unbound
    int jobs = 0;        // jobs_.size(), including terminal
    int live = 0;        // non-terminal jobs (active + queued)
    int max_queue = 8;
    int queue_timeout_seconds = 300;
    uint64_t admitted = 0, completed = 0, rejected = 0, timed_out = 0, cancelled = 0;
    uint64_t failed = 0; // rejected + timed_out + cancelled
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
    int live_count() const; // non-terminal jobs (active + queued)
    double job_wait_s(uint64_t id) const; // 0 if unknown
    int queue_timeout_s() const;
    uint64_t failed_count() const; // rejected + timed_out + cancelled
    uint64_t admitted_count() const; // snapshot.admitted
    uint64_t completed_count() const; // snapshot.completed
    uint64_t rejected_count() const; // snapshot.rejected
    uint64_t timed_out_count() const; // snapshot.timed_out
    uint64_t cancelled_count() const; // snapshot.cancelled
    int idle_count() const; // max(0, capacity - busy_slots)
    int capacity() const; // sessions n_slots, or 1 if unbound / n_slots < 1
    void snapshot(SchedulerSnapshot &out) const;

private:
    Engine *engine_ = nullptr;
    SessionStore *sessions_ = nullptr;
    int max_queue_ = 8;
    int queue_timeout_s_ = 300;
    uint64_t next_id_ = 1;
    std::vector<BatchJob> jobs_;
    uint64_t admitted_ = 0, completed_ = 0, rejected_ = 0, timed_out_ = 0, cancelled_ = 0;
};

} // namespace mvllm
