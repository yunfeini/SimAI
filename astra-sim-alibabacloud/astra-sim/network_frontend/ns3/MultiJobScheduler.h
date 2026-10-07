/*
 * Copyright (c) 2024, Alibaba Group
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef SIMAI_MULTI_JOB_SCHEDULER_H
#define SIMAI_MULTI_JOB_SCHEDULER_H

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace SimAiMultiJob {

enum class Policy { FirstFit, BestFit, FCFS, FragmentationGradient, Crux };

struct JobSpec {
  uint32_t job_id = 0;
  std::string workload;
  uint32_t num_gpus = 0;
  uint64_t submit_time_ns = 0;
  std::string name;
  uint64_t gpu_memory_bytes = 0;
  uint64_t host_memory_bytes = 0;
  uint64_t communication_bytes = 0;
};

struct GpuResource {
  uint64_t gpu_memory_capacity_bytes = 0;
  uint64_t gpu_memory_used_bytes = 0;
  uint64_t host_memory_capacity_bytes = 0;
  uint64_t host_memory_used_bytes = 0;
  bool reserved = false;
};

struct Placement {
  uint32_t job_id = 0;
  uint32_t num_gpus = 0;
  std::vector<uint32_t> gpu_ids;
};

class Scheduler {
 public:
  using StartCallback = std::function<void(const JobSpec&, const Placement&)>;
  using FinishCallback = std::function<void(const JobSpec&, const Placement&)>;

  Scheduler(uint32_t total_gpus, uint32_t gpus_per_server, Policy policy)
      : total_gpus_(total_gpus),
        gpus_per_server_(std::max(1u, gpus_per_server)),
        policy_(policy),
        resources_(total_gpus) {}

  Scheduler(const std::vector<GpuResource>& resources,
            uint32_t gpus_per_server, Policy policy)
      : total_gpus_(static_cast<uint32_t>(resources.size())),
        gpus_per_server_(std::max(1u, gpus_per_server)),
        policy_(policy),
        resources_(resources) {}

  void SetLogFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    log_.close();
    log_.open(path.c_str(), std::ios::out | std::ios::trunc);
    if (!log_) throw std::runtime_error("cannot open scheduler log: " + path);
    log_ << "event,time_ns,job_id,job_name,gpu_id,gpu_state,gpu_job_id,gpu_job_name,"
            "gpu_memory_used_bytes,gpu_memory_capacity_bytes,"
            "host_memory_used_bytes,host_memory_capacity_bytes,"
            "job_gpu_memory_bytes,job_host_memory_bytes,queue_latency_ns,"
            "service_latency_ns,e2e_latency_ns\n";
  }

  void SetGpuMemoryCapacity(uint32_t gpu_id, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gpu_id >= resources_.size()) {
      throw std::runtime_error("invalid GPU resource id");
    }
    resources_[gpu_id].gpu_memory_capacity_bytes = bytes;
  }

  void SetHostMemoryCapacity(uint32_t gpu_id, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gpu_id >= resources_.size()) {
      throw std::runtime_error("invalid GPU resource id");
    }
    resources_[gpu_id].host_memory_capacity_bytes = bytes;
  }

  const std::vector<GpuResource>& Resources() const { return resources_; }

  void SetStartCallback(StartCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    start_callback_ = std::move(callback);
  }

  void SetFinishCallback(FinishCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    finish_callback_ = std::move(callback);
  }

  void Submit(const JobSpec& job) {
    Submit(job, 0);
  }

  void Submit(const JobSpec& job, uint64_t now_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (job.num_gpus == 0 || job.num_gpus > total_gpus_) {
      throw std::runtime_error("job " + std::to_string(job.job_id) +
                               " requests an invalid GPU count");
    }
    if (std::any_of(pending_.begin(), pending_.end(), [&](const PendingJob& item) {
          return item.job.job_id == job.job_id;
        }) ||
        std::any_of(active_.begin(), active_.end(), [&](const ActiveJob& item) {
          return item.job.job_id == job.job_id;
        })) {
      throw std::runtime_error("duplicate job_id " +
                               std::to_string(job.job_id));
    }
    PendingJob pending;
    pending.job = job;
    pending.submit_time_ns = now_ns;
    pending_.push_back(pending);
    LogSnapshot("submit", now_ns, job.job_id, job.name, nullptr, 0, 0, 0);
    schedule_time_ns_ = now_ns;
    Schedule();
  }

  void Finish(uint32_t job_id) {
    Finish(job_id, 0);
  }

  void Finish(uint32_t job_id, uint64_t now_ns) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::find_if(active_.begin(), active_.end(),
                           [job_id](const ActiveJob& item) {
                             return item.job.job_id == job_id;
                           });
    if (it == active_.end()) {
      return;
    }
    for (uint32_t gpu_id : it->placement.gpu_ids) {
      if (gpu_id >= resources_.size()) continue;
      GpuResource& resource = resources_[gpu_id];
      resource.reserved = false;
      resource.gpu_memory_used_bytes =
          resource.gpu_memory_used_bytes >= it->job.gpu_memory_bytes
              ? resource.gpu_memory_used_bytes - it->job.gpu_memory_bytes
              : 0;
      resource.host_memory_used_bytes =
          resource.host_memory_used_bytes >= it->job.host_memory_bytes
              ? resource.host_memory_used_bytes - it->job.host_memory_bytes
              : 0;
    }
    const uint64_t queue_latency_ns =
        it->start_time_ns >= it->submit_time_ns
            ? it->start_time_ns - it->submit_time_ns
            : 0;
    const uint64_t service_latency_ns =
        now_ns >= it->start_time_ns ? now_ns - it->start_time_ns : 0;
    const uint64_t e2e_latency_ns =
        now_ns >= it->submit_time_ns ? now_ns - it->submit_time_ns : 0;
    JobSpec job = it->job;
    Placement placement = it->placement;
    active_.erase(it);
    LogSnapshot("finish", now_ns, job.job_id, job.name, &job,
                queue_latency_ns, service_latency_ns, e2e_latency_ns);
    if (finish_callback_) {
      finish_callback_(job, placement);
    }
    schedule_time_ns_ = now_ns;
    Schedule();
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.empty() && active_.empty();
  }

  size_t PendingJobs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_.size();
  }

  size_t ActiveJobs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_.size();
  }

  bool IsGpuReserved(uint32_t gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu_id >= resources_.size() || resources_[gpu_id].reserved;
  }

 private:
  struct PendingJob {
    JobSpec job;
    uint64_t submit_time_ns = 0;
  };

  struct ActiveJob {
    JobSpec job;
    Placement placement;
    uint64_t submit_time_ns = 0;
    uint64_t start_time_ns = 0;
  };

  void LogSnapshot(const std::string& event, uint64_t now_ns, uint32_t job_id,
                   const std::string& job_name, const JobSpec* job,
                   uint64_t queue_latency_ns, uint64_t service_latency_ns,
                   uint64_t e2e_latency_ns) {
    if (!log_) return;
    for (uint32_t gpu_id = 0; gpu_id < resources_.size(); ++gpu_id) {
      const GpuResource& resource = resources_[gpu_id];
      const JobSpec* owner = nullptr;
      for (const ActiveJob& active : active_) {
        if (std::find(active.placement.gpu_ids.begin(),
                      active.placement.gpu_ids.end(), gpu_id) !=
            active.placement.gpu_ids.end()) {
          owner = &active.job;
          break;
        }
      }
      log_ << event << ',' << now_ns << ',' << job_id << ',' << job_name << ','
           << gpu_id << ',' << (resource.reserved ? "busy" : "free") << ','
           << (owner ? std::to_string(owner->job_id) : "") << ','
           << (owner ? owner->name : "") << ','
           << resource.gpu_memory_used_bytes << ','
           << resource.gpu_memory_capacity_bytes << ','
           << resource.host_memory_used_bytes << ','
           << resource.host_memory_capacity_bytes << ','
           << (owner ? owner->gpu_memory_bytes : (job ? job->gpu_memory_bytes : 0))
           << ','
           << (owner ? owner->host_memory_bytes : (job ? job->host_memory_bytes : 0))
           << ',' << queue_latency_ns << ',' << service_latency_ns << ','
           << e2e_latency_ns << '\n';
    }
    log_.flush();
  }

  bool Fits(const GpuResource& resource, const JobSpec& job) const {
    if (resource.reserved) return false;
    if (job.gpu_memory_bytes != 0 &&
        (resource.gpu_memory_capacity_bytes == 0 ||
         resource.gpu_memory_capacity_bytes -
                 std::min(resource.gpu_memory_capacity_bytes,
                          resource.gpu_memory_used_bytes) <
             job.gpu_memory_bytes)) {
      return false;
    }
    if (job.host_memory_bytes != 0 &&
        (resource.host_memory_capacity_bytes == 0 ||
         resource.host_memory_capacity_bytes -
                 std::min(resource.host_memory_capacity_bytes,
                          resource.host_memory_used_bytes) <
             job.host_memory_bytes)) {
      return false;
    }
    return true;
  }

  bool FindPlacement(const JobSpec& job, Placement* placement) const {
    // SimCCL models a physical node as gpus_per_server_ ranks connected to one
    // NVSwitch.  Select complete servers, but do not require server IDs to be
    // contiguous (for example, [server 0, server 2]).
    if (gpus_per_server_ > 1 && job.num_gpus % gpus_per_server_ == 0) {
      const uint32_t server_count = total_gpus_ / gpus_per_server_;
      const uint32_t required_servers = job.num_gpus / gpus_per_server_;
      std::vector<uint32_t> servers;
      for (uint32_t server = 0; server < server_count; ++server) {
        bool eligible = true;
        for (uint32_t local = 0; local < gpus_per_server_; ++local) {
          if (!Fits(resources_[server * gpus_per_server_ + local], job)) {
            eligible = false;
            break;
          }
        }
        if (eligible) servers.push_back(server);
      }
      if (servers.size() >= required_servers) {
        if (policy_ == Policy::BestFit) {
          std::stable_sort(servers.begin(), servers.end(),
                           [this](uint32_t lhs, uint32_t rhs) {
            uint64_t lrem = std::numeric_limits<uint64_t>::max();
            uint64_t rrem = std::numeric_limits<uint64_t>::max();
            for (uint32_t local = 0; local < gpus_per_server_; ++local) {
              const GpuResource& l = resources_[lhs * gpus_per_server_ + local];
              const GpuResource& r = resources_[rhs * gpus_per_server_ + local];
              if (l.gpu_memory_capacity_bytes != 0) {
                lrem = std::min(lrem, l.gpu_memory_capacity_bytes -
                                          l.gpu_memory_used_bytes);
              }
              if (r.gpu_memory_capacity_bytes != 0) {
                rrem = std::min(rrem, r.gpu_memory_capacity_bytes -
                                          r.gpu_memory_used_bytes);
              }
            }
            return lrem < rrem;
          });
        }
        placement->num_gpus = job.num_gpus;
        placement->gpu_ids.clear();
        for (uint32_t i = 0; i < required_servers; ++i) {
          for (uint32_t local = 0; local < gpus_per_server_; ++local) {
            placement->gpu_ids.push_back(servers[i] * gpus_per_server_ + local);
          }
        }
        return true;
      }
      return false;
    }

    std::vector<uint32_t> candidates;
    for (uint32_t gpu_id = 0; gpu_id < total_gpus_; ++gpu_id) {
      if (Fits(resources_[gpu_id], job)) candidates.push_back(gpu_id);
    }
    if (candidates.size() < job.num_gpus) return false;
    if (policy_ == Policy::BestFit) {
      std::stable_sort(candidates.begin(), candidates.end(),
                       [this, &job](uint32_t lhs, uint32_t rhs) {
        const GpuResource& l = resources_[lhs];
        const GpuResource& r = resources_[rhs];
        uint64_t lrem = l.gpu_memory_capacity_bytes == 0
                            ? std::numeric_limits<uint64_t>::max()
                            : l.gpu_memory_capacity_bytes - l.gpu_memory_used_bytes;
        uint64_t rrem = r.gpu_memory_capacity_bytes == 0
                            ? std::numeric_limits<uint64_t>::max()
                            : r.gpu_memory_capacity_bytes - r.gpu_memory_used_bytes;
        return lrem < rrem;
      });
    }
    placement->num_gpus = job.num_gpus;
    placement->gpu_ids.assign(candidates.begin(), candidates.begin() + job.num_gpus);
    std::sort(placement->gpu_ids.begin(), placement->gpu_ids.end());
    return true;
  }

  uint64_t FragmentationScore(const std::vector<uint32_t>& selected) const {
    std::vector<bool> occupied(total_gpus_, false);
    for (uint32_t gpu_id : selected) {
      if (gpu_id < occupied.size()) occupied[gpu_id] = true;
    }
    uint32_t free_count = 0;
    uint32_t largest_run = 0;
    uint32_t run = 0;
    for (bool used : occupied) {
      if (!used) {
        ++free_count;
        ++run;
        largest_run = std::max(largest_run, run);
      } else {
        run = 0;
      }
    }
    if (free_count == 0) return 0;
    // Number of free GPUs outside the largest contiguous free interval.
    return free_count - largest_run;
  }

  bool FindFragmentationPlacement(const JobSpec& job,
                                  Placement* placement) const {
    std::vector<uint32_t> candidates;
    for (uint32_t gpu_id = 0; gpu_id < total_gpus_; ++gpu_id) {
      if (Fits(resources_[gpu_id], job)) candidates.push_back(gpu_id);
    }
    if (candidates.size() < job.num_gpus) return false;
    std::vector<uint32_t> best;
    uint64_t best_score = std::numeric_limits<uint64_t>::max();
    // The scheduler is intended for moderate GPU counts; enumerate contiguous
    // windows first, then use a deterministic sparse fallback.
    for (uint32_t begin = 0; begin + job.num_gpus <= total_gpus_; ++begin) {
      std::vector<uint32_t> selected;
      bool valid = true;
      for (uint32_t i = 0; i < job.num_gpus; ++i) {
        const uint32_t gpu_id = begin + i;
        if (!Fits(resources_[gpu_id], job)) {
          valid = false;
          break;
        }
        selected.push_back(gpu_id);
      }
      if (!valid) continue;
      const uint64_t score = FragmentationScore(selected);
      if (score < best_score) {
        best_score = score;
        best = selected;
      }
    }
    if (best.empty()) {
      best.assign(candidates.begin(), candidates.begin() + job.num_gpus);
    }
    placement->num_gpus = job.num_gpus;
    placement->gpu_ids = best;
    return true;
  }

  bool FindCruxPlacement(const JobSpec& job, Placement* placement) const {
    // Crux-style priority: communication-heavy and GPU-heavy jobs receive
    // priority at the queue level; placement minimizes cross-server edges.
    std::vector<uint32_t> candidates;
    for (uint32_t gpu_id = 0; gpu_id < total_gpus_; ++gpu_id) {
      if (Fits(resources_[gpu_id], job)) candidates.push_back(gpu_id);
    }
    if (candidates.size() < job.num_gpus) return false;
    std::vector<uint32_t> best;
    uint32_t best_span = std::numeric_limits<uint32_t>::max();
    for (uint32_t begin = 0; begin < total_gpus_; ++begin) {
      std::vector<uint32_t> window;
      for (uint32_t gpu_id : candidates) {
        if (gpu_id >= begin &&
            gpu_id < begin + job.num_gpus + gpus_per_server_ - 1) {
          window.push_back(gpu_id);
        }
      }
      if (window.size() < job.num_gpus) continue;
      window.resize(job.num_gpus);
      const uint32_t span = window.back() / gpus_per_server_ -
                            window.front() / gpus_per_server_;
      if (span < best_span) {
        best_span = span;
        best = window;
      }
    }
    if (best.empty()) {
      best.assign(candidates.begin(), candidates.begin() + job.num_gpus);
    }
    placement->num_gpus = job.num_gpus;
    placement->gpu_ids = best;
    return true;
  }

  double CruxPriority(const PendingJob& pending) const {
    const double communication = static_cast<double>(pending.job.communication_bytes);
    const double gpu_count = static_cast<double>(pending.job.num_gpus);
    const double wait = schedule_time_ns_ >= pending.submit_time_ns
                            ? static_cast<double>(schedule_time_ns_ - pending.submit_time_ns)
                            : 0.0;
    return communication * std::max(1.0, gpu_count) + wait;
  }

  void Schedule() {
    if (policy_ == Policy::FCFS) {
      ScheduleFcfs();
      return;
    }
    if (policy_ == Policy::FragmentationGradient) {
      ScheduleFragmentationGradient();
      return;
    }
    if (policy_ == Policy::Crux) {
      ScheduleCrux();
      return;
    }
    for (auto it = pending_.begin(); it != pending_.end();) {
      Placement placement;
      placement.job_id = it->job.job_id;
      if (!FindPlacement(it->job, &placement)) {
        ++it;
        continue;
      }
      for (uint32_t gpu_id : placement.gpu_ids) {
        resources_[gpu_id].reserved = true;
        resources_[gpu_id].gpu_memory_used_bytes += it->job.gpu_memory_bytes;
        resources_[gpu_id].host_memory_used_bytes += it->job.host_memory_bytes;
      }
      ActiveJob active;
      active.job = it->job;
      active.placement = placement;
      active.submit_time_ns = it->submit_time_ns;
      active.start_time_ns = schedule_time_ns_;
      it = pending_.erase(it);
      active_.push_back(active);
      LogSnapshot("schedule", schedule_time_ns_, active.job.job_id,
                  active.job.name, &active.job,
                  active.start_time_ns >= active.submit_time_ns
                      ? active.start_time_ns - active.submit_time_ns
                      : 0,
                  0, 0);
      if (start_callback_) {
        start_callback_(active.job, active.placement);
      }
    }
  }

  void Activate(const PendingJob& pending, const Placement& placement) {
    for (uint32_t gpu_id : placement.gpu_ids) {
      resources_[gpu_id].reserved = true;
      resources_[gpu_id].gpu_memory_used_bytes += pending.job.gpu_memory_bytes;
      resources_[gpu_id].host_memory_used_bytes += pending.job.host_memory_bytes;
    }
    ActiveJob active;
    active.job = pending.job;
    active.placement = placement;
    active.submit_time_ns = pending.submit_time_ns;
    active.start_time_ns = schedule_time_ns_;
    active_.push_back(active);
    LogSnapshot("schedule", schedule_time_ns_, active.job.job_id,
                active.job.name, &active.job,
                active.start_time_ns >= active.submit_time_ns
                    ? active.start_time_ns - active.submit_time_ns : 0,
                0, 0);
    if (start_callback_) start_callback_(active.job, active.placement);
  }

  void ScheduleFragmentationGradient() {
    for (auto it = pending_.begin(); it != pending_.end();) {
      Placement placement;
      placement.job_id = it->job.job_id;
      if (!FindFragmentationPlacement(it->job, &placement)) {
        ++it;
        continue;
      }
      PendingJob pending = *it;
      it = pending_.erase(it);
      Activate(pending, placement);
    }
  }

  void ScheduleCrux() {
    while (!pending_.empty()) {
      auto best_it = pending_.end();
      Placement best_placement;
      double best_priority = -1.0;
      for (auto it = pending_.begin(); it != pending_.end(); ++it) {
        Placement placement;
        placement.job_id = it->job.job_id;
        if (!FindCruxPlacement(it->job, &placement)) continue;
        const double priority = CruxPriority(*it);
        if (best_it == pending_.end() || priority > best_priority) {
          best_it = it;
          best_priority = priority;
          best_placement = placement;
        }
      }
      if (best_it == pending_.end()) return;
      PendingJob pending = *best_it;
      pending_.erase(best_it);
      Activate(pending, best_placement);
    }
  }

  void ScheduleFcfs() {
    while (!pending_.empty()) {
      auto it = pending_.begin();
      Placement placement;
      placement.job_id = it->job.job_id;
      if (!FindPlacement(it->job, &placement)) {
        // Strict FCFS: a blocked head-of-line job prevents later jobs from
        // overtaking it, even if they would fit the currently free GPUs.
        return;
      }
      for (uint32_t gpu_id : placement.gpu_ids) {
        resources_[gpu_id].reserved = true;
        resources_[gpu_id].gpu_memory_used_bytes += it->job.gpu_memory_bytes;
        resources_[gpu_id].host_memory_used_bytes += it->job.host_memory_bytes;
      }
      ActiveJob active;
      active.job = it->job;
      active.placement = placement;
      active.submit_time_ns = it->submit_time_ns;
      active.start_time_ns = schedule_time_ns_;
      pending_.erase(it);
      active_.push_back(active);
      LogSnapshot("schedule", schedule_time_ns_, active.job.job_id,
                  active.job.name, &active.job,
                  active.start_time_ns >= active.submit_time_ns
                      ? active.start_time_ns - active.submit_time_ns
                      : 0,
                  0, 0);
      if (start_callback_) {
        start_callback_(active.job, active.placement);
      }
    }
  }

  uint32_t total_gpus_;
  uint32_t gpus_per_server_;
  Policy policy_;
  std::vector<GpuResource> resources_;
  std::vector<PendingJob> pending_;
  std::vector<ActiveJob> active_;
  StartCallback start_callback_;
  FinishCallback finish_callback_;
  mutable std::mutex mutex_;
  std::ofstream log_;
  uint64_t schedule_time_ns_ = 0;
};

inline std::string Trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() &&
         std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  size_t end = value.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

inline std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (char ch : line) {
    if (ch == '"') {
      quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      fields.push_back(Trim(field));
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(Trim(field));
  return fields;
}

inline std::vector<JobSpec> LoadJobs(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open job file: " + path);
  }
  std::vector<JobSpec> jobs;
  std::string line;
  uint32_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::vector<std::string> fields = SplitCsv(line);
    if (!fields.empty() && fields[0] == "job_id") {
      continue;
    }
    if (fields.size() < 4 || fields.size() > 8) {
      throw std::runtime_error("invalid job file line " +
                               std::to_string(line_number));
    }
    JobSpec job;
    try {
      job.job_id = static_cast<uint32_t>(std::stoul(fields[0]));
      job.workload = fields[1];
      job.num_gpus = static_cast<uint32_t>(std::stoul(fields[2]));
      job.submit_time_ns = std::stoull(fields[3]);
    } catch (const std::exception&) {
      throw std::runtime_error("invalid value in job file line " +
                               std::to_string(line_number));
    }
    job.name = fields.size() >= 5 && !fields[4].empty()
                   ? fields[4]
                   : "job-" + std::to_string(job.job_id);
    try {
      if (fields.size() >= 6 && !fields[5].empty()) {
        job.gpu_memory_bytes = std::stoull(fields[5]);
      }
      if (fields.size() >= 7 && !fields[6].empty()) {
        job.host_memory_bytes = std::stoull(fields[6]);
      }
      if (fields.size() >= 8 && !fields[7].empty()) {
        job.communication_bytes = std::stoull(fields[7]);
      }
    } catch (const std::exception&) {
      throw std::runtime_error("invalid resource value in job file line " +
                               std::to_string(line_number));
    }
    if (job.workload.empty()) {
      throw std::runtime_error("empty workload in job file line " +
                               std::to_string(line_number));
    }
    if (std::any_of(jobs.begin(), jobs.end(), [&](const JobSpec& item) {
          return item.job_id == job.job_id;
        })) {
      throw std::runtime_error("duplicate job_id " +
                               std::to_string(job.job_id));
    }
    jobs.push_back(job);
  }
  std::stable_sort(jobs.begin(), jobs.end(),
                   [](const JobSpec& lhs, const JobSpec& rhs) {
                     return lhs.submit_time_ns < rhs.submit_time_ns;
                   });
  return jobs;
}

inline std::vector<GpuResource> LoadGpuResources(
    const std::string& path, uint32_t total_gpus,
    uint64_t default_gpu_memory = 0, uint64_t default_host_memory = 0) {
  std::vector<GpuResource> resources(total_gpus);
  for (GpuResource& resource : resources) {
    resource.gpu_memory_capacity_bytes = default_gpu_memory;
    resource.host_memory_capacity_bytes = default_host_memory;
  }
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open GPU resource file: " + path);
  }
  std::string line;
  uint32_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> fields = SplitCsv(line);
    if (!fields.empty() && fields[0] == "gpu_id") continue;
    if (fields.size() != 3) {
      throw std::runtime_error("invalid GPU resource file line " +
                               std::to_string(line_number));
    }
    try {
      const uint32_t gpu_id = static_cast<uint32_t>(std::stoul(fields[0]));
      if (gpu_id >= total_gpus) {
        throw std::runtime_error("GPU resource id is out of range at line " +
                                 std::to_string(line_number));
      }
      resources[gpu_id].gpu_memory_capacity_bytes = std::stoull(fields[1]);
      resources[gpu_id].host_memory_capacity_bytes = std::stoull(fields[2]);
    } catch (const std::exception&) {
      throw std::runtime_error("invalid GPU resource value at line " +
                               std::to_string(line_number));
    }
  }
  return resources;
}

inline Policy ParsePolicy(const std::string& value) {
  if (value == "fcfs" || value == "fifo") {
    return Policy::FCFS;
  }
  if (value == "fgd" || value == "fragmentation-gradient" ||
      value == "fragmentation_gradient") {
    return Policy::FragmentationGradient;
  }
  if (value == "crux") {
    return Policy::Crux;
  }
  if (value == "first-fit" || value == "first_fit") {
    return Policy::FirstFit;
  }
  if (value == "best-fit" || value == "best_fit") {
    return Policy::BestFit;
  }
  throw std::runtime_error("unknown scheduling policy: " + value);
}

}  // namespace SimAiMultiJob

#endif
