#include "RdmaFlowRegistry.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef NDEBUG
#error "Compile this regression test with assertions enabled"
#endif

namespace {

struct Tag {
  uint32_t job_id = 0;
  std::vector<int> child_flows;
};

using Registry = SimAiNs3::RdmaFlowRegistry<Tag>;
using Side = SimAiNs3::CompletionSide;
using Status = SimAiNs3::CompletionStatus;

void TestCompletionOrders() {
  for (Side first : {Side::Send, Side::Receive}) {
    Registry registry;
    Tag result;
    assert(registry.Register(46, 60, 10000, Tag{294, {7, 9}}));
    assert(registry.Complete(46, 60, 10000, first, result) == Status::Recorded);
    assert(result.job_id == 294 && result.child_flows == std::vector<int>({7, 9}));
    assert(registry.Size() == 1);
    assert(registry.Complete(46, 60, 10000, first, result) == Status::Duplicate);
    assert(registry.Size() == 1);
    const Side second = first == Side::Send ? Side::Receive : Side::Send;
    assert(registry.Complete(46, 60, 10000, second, result) == Status::Recorded);
    assert(registry.Size() == 0);
    // The callback's tag remains valid after the registry releases its copy.
    assert(result.job_id == 294 && result.child_flows == std::vector<int>({7, 9}));
    assert(registry.Complete(46, 60, 10000, second, result) == Status::Missing);
  }
}

void TestJobIsolationAndPortReuse() {
  Registry registry;
  Tag result;
  assert(registry.Register(46, 60, 10000, Tag{294, {1}}));
  assert(registry.Register(46, 61, 10000, Tag{295, {2}}));
  assert(registry.Register(60, 46, 10000, Tag{296, {3}}));
  assert(registry.Register(65536 + 46, 60, 10000, Tag{297, {4}}));
  assert(!registry.Register(46, 60, 10000, Tag{999, {5}}));
  assert(registry.Complete(46, 60, 10000, Side::Send, result) == Status::Recorded);
  assert(result.job_id == 294);
  assert(registry.Complete(46, 60, 10000, Side::Receive, result) == Status::Recorded);
  assert(registry.Register(46, 60, 10000, Tag{300, {6}}));
  assert(registry.Complete(46, 60, 10000, Side::Receive, result) == Status::Recorded);
  assert(result.job_id == 300);
  assert(registry.Complete(46, 60, 10000, Side::Send, result) == Status::Recorded);
  assert(registry.Complete(46, 61, 10000, Side::Receive, result) == Status::Recorded);
  assert(result.job_id == 295);
  assert(registry.Complete(60, 46, 10000, Side::Send, result) == Status::Recorded);
  assert(result.job_id == 296);
  assert(registry.Complete(65536 + 46, 60, 10000, Side::Receive, result) == Status::Recorded);
  assert(result.job_id == 297);
  assert(registry.Size() == 3);
}

void TestPortCounterWrap() {
  Registry registry;
  Tag result;
  uint16_t next_port = std::numeric_limits<uint16_t>::max();
  assert(registry.Register(46, 60, next_port, Tag{1, {}}));
  assert(registry.Register(46, 60, 0, Tag{2, {}}));
  assert(registry.AllocatePort(46, 60, next_port, Tag{3, {}}) == 1);
  assert(next_port == 2);
  assert(registry.Complete(46, 60, 65535, Side::Send, result) == Status::Recorded);
  assert(result.job_id == 1);
  assert(registry.Complete(46, 60, 0, Side::Send, result) == Status::Recorded);
  assert(result.job_id == 2);
}

void TestLargeTraceCleanup() {
  Registry registry;
  Tag result;
  // Keep one incomplete flow while a large history of other flows completes.
  assert(registry.Register(64, 120, 9000, Tag{1, {}}));
  uint16_t next_port = 10000;
  for (uint32_t job = 2; job < 200002; ++job) {
    const uint16_t port = registry.AllocatePort(46, 60, next_port, Tag{job, {1}});
    const Side first = job % 2 ? Side::Send : Side::Receive;
    const Side second = first == Side::Send ? Side::Receive : Side::Send;
    assert(registry.Complete(46, 60, port, first, result) == Status::Recorded);
    assert(result.job_id == job);
    assert(registry.Complete(46, 60, port, second, result) == Status::Recorded);
    assert(registry.Size() == 1);
  }
  assert(registry.Complete(64, 120, 9000, Side::Receive, result) == Status::Recorded);
  assert(registry.Complete(64, 120, 9000, Side::Send, result) == Status::Recorded);
  assert(registry.Size() == 0);
}

// Mirror the MTP primitive's ownership contract: its destructor does nothing.
struct ManualCriticalSection {
  static bool held;
  static unsigned acquisitions;
  static unsigned releases;

  ManualCriticalSection() {
    assert(!held);
    held = true;
    ++acquisitions;
  }
  ~ManualCriticalSection() = default;
  void ExitSection() {
    assert(held);
    held = false;
    ++releases;
  }
};

bool ManualCriticalSection::held = false;
unsigned ManualCriticalSection::acquisitions = 0;
unsigned ManualCriticalSection::releases = 0;
using Guard = SimAiNs3::ScopedCriticalSection<ManualCriticalSection>;

void CompleteMissingFlow(Registry& registry) {
  Guard guard;
  Tag result;
  if (registry.Complete(46, 60, 10000, Side::Send, result) != Status::Recorded) {
    return;
  }
  assert(false);
}

void TestLockRelease() {
  Registry registry;
  CompleteMissingFlow(registry);
  assert(!ManualCriticalSection::held);
  {
    Guard first;
    first.ExitSection();
    {
      Guard second;
      first.ExitSection();
      assert(ManualCriticalSection::held);
    }
    assert(!ManualCriticalSection::held);
  }
  // Exhaustion must release the lock without replacing an existing flow.
  for (uint32_t port = 0; port <= std::numeric_limits<uint16_t>::max(); ++port) {
    assert(registry.Register(46, 60, static_cast<uint16_t>(port), Tag{port + 1, {}}));
  }
  bool threw = false;
  try {
    Guard guard;
    uint16_t next_port = 10000;
    registry.AllocatePort(46, 60, next_port, Tag{999999, {}});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw && !ManualCriticalSection::held);
  assert(registry.Size() == 65536);
  assert(ManualCriticalSection::acquisitions == ManualCriticalSection::releases);
}

}  // namespace

int main() {
  TestCompletionOrders();
  TestJobIsolationAndPortReuse();
  TestPortCounterWrap();
  TestLargeTraceCleanup();
  TestLockRelease();
  std::cout << "RDMA flow registry and lock regression checks passed\n";
}
