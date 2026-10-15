#ifndef SIMAI_NS3_RDMA_FLOW_REGISTRY_H
#define SIMAI_NS3_RDMA_FLOW_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace SimAiNs3 {

// Adapt the MTP lock, whose destructor does not release it. Explicit early
// release is still allowed, but leaving a scope releases the lock exactly once.
template <typename CriticalSection>
class ScopedCriticalSection {
 public:
  ScopedCriticalSection() = default;
  ScopedCriticalSection(const ScopedCriticalSection&) = delete;
  ScopedCriticalSection& operator=(const ScopedCriticalSection&) = delete;
  ~ScopedCriticalSection() { ExitSection(); }

  void ExitSection() {
    if (locked_) {
      locked_ = false;
      section_.ExitSection();
    }
  }

 private:
  CriticalSection section_;
  bool locked_ = true;
};

enum class CompletionSide { Send, Receive };
enum class CompletionStatus { Recorded, Missing, Duplicate };

// ns-3 completion callbacks expose physical endpoints and a source port, not
// a job ID. Ports must be unique among live flows on that endpoint pair. Keep
// the job ID in Tag so collective callbacks remain isolated between jobs.
// The caller holds the MTP critical section for every access to this registry.
template <typename Tag>
class RdmaFlowRegistry {
 public:
  bool Register(uint32_t src, uint32_t dst, uint16_t port, const Tag& tag) {
    return flows_.emplace(PortKey{src, dst, port}, FlowState{tag}).second;
  }

  uint16_t AllocatePort(uint32_t src, uint32_t dst, uint16_t& next_port,
                        const Tag& tag) {
    for (uint32_t attempt = 0;
         attempt <= std::numeric_limits<uint16_t>::max(); ++attempt) {
      const uint16_t port = next_port++;
      if (Register(src, dst, port, tag)) {
        return port;
      }
    }
    throw std::runtime_error("all RDMA source ports are in use for GPU pair " +
                             std::to_string(src) + " -> " + std::to_string(dst));
  }

  CompletionStatus Complete(uint32_t src, uint32_t dst, uint16_t port,
                            CompletionSide side, Tag& tag) {
    auto it = flows_.find(PortKey{src, dst, port});
    if (it == flows_.end()) {
      return CompletionStatus::Missing;
    }
    bool& finished = side == CompletionSide::Send
                         ? it->second.send_finished
                         : it->second.receive_finished;
    if (finished) {
      return CompletionStatus::Duplicate;
    }
    tag = it->second.tag;
    finished = true;
    if (it->second.send_finished && it->second.receive_finished) {
      flows_.erase(it);
    }
    return CompletionStatus::Recorded;
  }

  std::size_t Size() const { return flows_.size(); }

 private:
  struct PortKey {
    uint32_t src;
    uint32_t dst;
    uint16_t port;

    bool operator==(const PortKey& other) const {
      return src == other.src && dst == other.dst && port == other.port;
    }
  };

  struct PortKeyHash {
    std::size_t operator()(const PortKey& key) const {
      const uint64_t endpoints = (static_cast<uint64_t>(key.src) << 32) | key.dst;
      const std::size_t hash = std::hash<uint64_t>{}(endpoints);
      return hash ^ (std::hash<uint16_t>{}(key.port) + 0x9e3779b9U +
                     (hash << 6) + (hash >> 2));
    }
  };

  struct FlowState {
    Tag tag;
    bool send_finished = false;
    bool receive_finished = false;
  };

  std::unordered_map<PortKey, FlowState, PortKeyHash> flows_;
};

}  // namespace SimAiNs3

#endif  // SIMAI_NS3_RDMA_FLOW_REGISTRY_H
