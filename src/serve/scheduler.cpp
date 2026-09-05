#include "scheduler.hpp"
#include "../engine.hpp"

#include <chrono>
#include <mutex>

namespace mvllm {
namespace {

std::mutex g_batch_mu;

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

bool terminal(BatchJobState s) {
    return s == BatchJobState::Done || s == BatchJobState::Cancelled;
}

int live_count(const std::vector<BatchJob> &jobs) {
    int n = 0;
    for (const auto &j : jobs)
        if (!terminal(j.state))
            ++n;
    return n;
}

BatchJob *find_job(std::vector<BatchJob> &jobs, uint64_t id) {
    for (auto &j : jobs)
        if (j.id == id)
            return &j;
    return nullptr;
}

const BatchJob *find_job(const std::vector<BatchJob> &jobs, uint64_t id) {
    for (const auto &j : jobs)
        if (j.id == id)
            return &j;
    return nullptr;
}

void commit_and_release(SessionStore *sessions, BatchJob &j) {
    if (!sessions)
        return;
    std::vector<int> hist = j.ids;
    hist.insert(hist.end(), j.out.tokens.begin(), j.out.tokens.end());
    sessions->commit(j.slot, hist);
    sessions->release(j.slot);
}

void end_slot(Engine *engine, int slot) {
    if (!engine)
        return;
    FamilyEngine *fe = engine->family_impl();
    if (fe)
        fe->end_generate(slot);
}

} // namespace

void BatchScheduler::bind(Engine *engine, SessionStore *sessions) {
    engine_ = engine;
    sessions_ = sessions;
}

void BatchScheduler::configure(int max_queue, int queue_timeout_s) {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    max_queue_ = max_queue > 0 ? max_queue : 8;
    queue_timeout_s_ = queue_timeout_s > 0 ? queue_timeout_s : 300;
}

uint64_t BatchScheduler::submit(int slot, const std::vector<int> &ids, const GenParams &gp,
                                std::string &err) {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    if (!engine_ || !sessions_) {
        err = "scheduler not bound";
        return 0;
    }
    if (::mvllm::live_count(jobs_) >= max_queue_) {
        ++rejected_;
        err = "queue full";
        return 0;
    }
    if (!sessions_->try_acquire(slot)) {
        err = "SLOT_BUSY";
        return 0;
    }
    BatchJob job;
    job.id = next_id_++;
    job.slot = slot;
    job.ids = ids;
    job.gp = gp;
    job.queued_at = now_s();
    // Official KvPrefix reuse is all-or-nothing; no LCP fallback.
    int official = engine_->prefix_match(slot, ids);
    if (official > 0)
        job.reuse = official;
    else if (gp.prefix_reuse > 0)
        job.reuse = gp.prefix_reuse;
    else
        job.reuse = 0;
    job.state = BatchJobState::Queued;
    job.status = Status::Ok;
    job.out.prompt_tokens = static_cast<int>(ids.size());
    jobs_.push_back(std::move(job));
    err.clear();
    return jobs_.back().id;
}

bool BatchScheduler::stop(uint64_t id) {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    BatchJob *j = find_job(jobs_, id);
    if (!j)
        return false;
    if (j->state == BatchJobState::Queued || j->state == BatchJobState::Running) {
        j->state = BatchJobState::Stopped;
        return true;
    }
    return j->state == BatchJobState::Stopped;
}

bool BatchScheduler::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    BatchJob *j = find_job(jobs_, id);
    if (!j || terminal(j->state))
        return false;
    end_slot(engine_, j->slot);
    if (sessions_)
        sessions_->release(j->slot);
    j->state = BatchJobState::Cancelled;
    j->err = "CANCELLED";
    ++cancelled_;
    return true;
}

int BatchScheduler::pump() {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    if (!engine_ || !sessions_)
        return ::mvllm::live_count(jobs_);

    FamilyEngine *fe = engine_->family_impl();

    auto finish_done = [&](BatchJob &j) {
        end_slot(engine_, j.slot);
        commit_and_release(sessions_, j);
        std::vector<int> hist = j.ids;
        hist.insert(hist.end(), j.out.tokens.begin(), j.out.tokens.end());
        engine_->prefix_commit(j.slot, hist);
        std::string perr;
        engine_->persist_commit(j.slot, hist, perr); // no-op if persist closed
        j.state = BatchJobState::Done;
        if (j.status != Status::Ok && j.err.empty())
            j.status = Status::Ok;
        ++completed_;
    };
    auto finish_err = [&](BatchJob &j, Status st, const std::string &e) {
        end_slot(engine_, j.slot);
        if (sessions_)
            sessions_->release(j.slot);
        j.state = BatchJobState::Done;
        j.status = st;
        j.err = e;
    };

    for (auto &j : jobs_) {
        if (j.state == BatchJobState::Stopped)
            finish_done(j);
    }

    // Expire queued jobs that waited past queue_timeout_s_.
    if (queue_timeout_s_ > 0) {
        const double now = now_s();
        for (auto &j : jobs_) {
            if (j.state != BatchJobState::Queued)
                continue;
            if (now - j.queued_at < static_cast<double>(queue_timeout_s_))
                continue;
            end_slot(engine_, j.slot);
            if (sessions_)
                sessions_->release(j.slot);
            j.state = BatchJobState::Done;
            j.status = Status::InvalidArgument;
            j.err = "queue timeout";
            ++timed_out_;
        }
    }

    BatchJob *queued = nullptr;
    for (auto &j : jobs_) {
        if (j.state == BatchJobState::Queued) {
            queued = &j;
            break;
        }
    }

    if (queued) {
        queued->state = BatchJobState::Running;
        const double wait = now_s() - queued->queued_at;
        queued->wait_s = wait < 0.0 ? 0.0 : wait;
        ++admitted_;
        GenParams gp = queued->gp;
        gp.prefix_reuse = queued->reuse;
        gp.cache_slot = queued->slot;
        int applied = queued->reuse;
        std::string err;
        Status st = Status::Unsupported;
        if (fe)
            st = fe->begin_generate(queued->slot, queued->ids, gp, applied, err);
        if (st == Status::Unsupported) {
            err.clear();
            st = engine_->generate_ids(queued->ids, gp, queued->out, err);
            queued->reuse = 0;
            if (st != Status::Ok)
                finish_err(*queued, st, err);
            else
                finish_done(*queued);
            return ::mvllm::live_count(jobs_);
        }
        queued->reuse = applied;
        if (st != Status::Ok) {
            finish_err(*queued, st, err);
            return ::mvllm::live_count(jobs_);
        }
        queued->out.prompt_tokens = static_cast<int>(queued->ids.size());
        if (queued->state == BatchJobState::Stopped)
            finish_done(*queued);
        return ::mvllm::live_count(jobs_);
    }

    if (fe) {
        std::vector<BatchJob *> running;
        running.reserve(jobs_.size());
        for (auto &j : jobs_) {
            if (j.state != BatchJobState::Running)
                continue;
            const int max_new = j.gp.max_new_tokens;
            if (max_new <= 0 ||
                (max_new > 0 && static_cast<int>(j.out.tokens.size()) >= max_new)) {
                finish_done(j);
                continue;
            }
            running.push_back(&j);
        }

        bool use_loop = running.size() < 2;
        if (!use_loop) {
            const int n = static_cast<int>(running.size());
            std::vector<int> slots(static_cast<size_t>(n));
            std::vector<int> toks(static_cast<size_t>(n), 0);
            std::vector<uint8_t> dones(static_cast<size_t>(n), 0);
            for (int i = 0; i < n; ++i)
                slots[static_cast<size_t>(i)] = running[static_cast<size_t>(i)]->slot;
            std::string err;
            Status st = fe->next_tokens(slots.data(), n, toks.data(), dones.data(), err);
            if (st == Status::Unsupported) {
                use_loop = true;
            } else if (st != Status::Ok) {
                for (BatchJob *jp : running) {
                    if (jp->state == BatchJobState::Cancelled)
                        continue;
                    finish_err(*jp, st, err);
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    BatchJob &j = *running[static_cast<size_t>(i)];
                    if (j.state == BatchJobState::Cancelled)
                        continue;
                    j.out.tokens.push_back(toks[static_cast<size_t>(i)]);
                    ++j.out.completion_tokens;
                    if (j.gp.on_token)
                        j.gp.on_token(toks[static_cast<size_t>(i)]);
                    if (j.state == BatchJobState::Cancelled)
                        continue;
                    const int max_new = j.gp.max_new_tokens;
                    if (dones[static_cast<size_t>(i)] || j.state == BatchJobState::Stopped ||
                        (max_new > 0 && static_cast<int>(j.out.tokens.size()) >= max_new))
                        finish_done(j);
                }
            }
        }

        if (use_loop) {
            for (auto &j : jobs_) {
                if (j.state != BatchJobState::Running)
                    continue;
                const int max_new = j.gp.max_new_tokens;
                if (max_new <= 0 ||
                    (max_new > 0 && static_cast<int>(j.out.tokens.size()) >= max_new)) {
                    finish_done(j);
                    continue;
                }
                int tok = 0;
                bool done = false;
                std::string err;
                Status st = fe->next_token(j.slot, tok, done, err);
                if (j.state == BatchJobState::Cancelled)
                    continue;
                if (st != Status::Ok) {
                    finish_err(j, st, err);
                    continue;
                }
                j.out.tokens.push_back(tok);
                ++j.out.completion_tokens;
                if (j.gp.on_token)
                    j.gp.on_token(tok);
                if (j.state == BatchJobState::Cancelled)
                    continue;
                if (done || j.state == BatchJobState::Stopped ||
                    (max_new > 0 && static_cast<int>(j.out.tokens.size()) >= max_new))
                    finish_done(j);
            }
        }
    }

    for (auto &j : jobs_) {
        if (j.state == BatchJobState::Stopped)
            finish_done(j);
    }

    return ::mvllm::live_count(jobs_);
}

bool BatchScheduler::finished(uint64_t id) const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    const BatchJob *j = find_job(jobs_, id);
    return !j || terminal(j->state);
}

const BatchJob *BatchScheduler::job(uint64_t id) const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return find_job(jobs_, id);
}

int BatchScheduler::queue_depth() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return ::mvllm::live_count(jobs_);
}

int BatchScheduler::running_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    int n = 0;
    for (const auto &j : jobs_)
        if (j.state == BatchJobState::Running || j.state == BatchJobState::Stopped)
            ++n;
    return n;
}

int BatchScheduler::queued_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    int n = 0;
    for (const auto &j : jobs_)
        if (j.state == BatchJobState::Queued)
            ++n;
    return n;
}

int BatchScheduler::max_queue() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return max_queue_;
}

int BatchScheduler::n_jobs() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return static_cast<int>(jobs_.size());
}

int BatchScheduler::live_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return ::mvllm::live_count(jobs_);
}

double BatchScheduler::job_wait_s(uint64_t id) const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    const BatchJob *j = find_job(jobs_, id);
    return j ? j->wait_s : 0.0;
}

int BatchScheduler::queue_timeout_s() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return queue_timeout_s_;
}

uint64_t BatchScheduler::failed_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return rejected_ + timed_out_ + cancelled_;
}

uint64_t BatchScheduler::admitted_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return admitted_;
}

uint64_t BatchScheduler::completed_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return completed_;
}

uint64_t BatchScheduler::rejected_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return rejected_;
}

uint64_t BatchScheduler::timed_out_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return timed_out_;
}

uint64_t BatchScheduler::cancelled_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    return cancelled_;
}

int BatchScheduler::idle_count() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    int cap = 1;
    if (sessions_) {
        cap = sessions_->n_slots();
        if (cap < 1)
            cap = 1;
    }
    int busy_slots = sessions_ ? sessions_->busy_count() : 0;
    return cap > busy_slots ? cap - busy_slots : 0;
}

int BatchScheduler::capacity() const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    int cap = 1;
    if (sessions_) {
        cap = sessions_->n_slots();
        if (cap < 1)
            cap = 1;
    }
    return cap;
}

void BatchScheduler::snapshot(SchedulerSnapshot &out) const {
    std::lock_guard<std::mutex> lock(g_batch_mu);
    int running = 0;
    int queued = 0;
    for (const auto &j : jobs_) {
        if (j.state == BatchJobState::Running || j.state == BatchJobState::Stopped)
            ++running;
        else if (j.state == BatchJobState::Queued)
            ++queued;
    }
    int cap = 1;
    if (sessions_) {
        cap = sessions_->n_slots();
        if (cap < 1)
            cap = 1;
    }
    out.active = running;
    out.queued = queued;
    out.live = out.active + out.queued;
    out.capacity = cap;
    out.busy_slots = sessions_ ? sessions_->busy_count() : 0;
    out.hist_tokens = sessions_ ? sessions_->history_total() : 0;
    out.idle = out.capacity > out.busy_slots ? out.capacity - out.busy_slots : 0;
    out.jobs = static_cast<int>(jobs_.size());
    out.max_queue = max_queue_;
    out.queue_timeout_seconds = queue_timeout_s_;
    out.admitted = admitted_;
    out.completed = completed_;
    out.rejected = rejected_;
    out.timed_out = timed_out_;
    out.cancelled = cancelled_;
    out.failed = out.rejected + out.timed_out + out.cancelled;
}

} // namespace mvllm
