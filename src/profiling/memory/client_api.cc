/*
 * Copyright (C) 2020 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "perfetto/profiling/memory/heap_profile.h"
#include "src/profiling/memory/heap_profile_internal.h"

#include <inttypes.h>
#include <malloc.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

#include "perfetto/base/build_config.h"
#include "perfetto/base/logging.h"
#include "perfetto/ext/base/no_destructor.h"
#include "perfetto/ext/base/unix_socket.h"
#include "perfetto/ext/base/utils.h"

#include "src/profiling/common/proc_utils.h"
#include "src/profiling/memory/client.h"
#include "src/profiling/memory/client_api_factory.h"
#include "src/profiling/memory/scoped_spinlock.h"
#include "src/profiling/memory/unhooked_allocator.h"
#include "src/profiling/memory/wire_protocol.h"

struct AHeapInfo {
  // Fields set by user.
  char heap_name[HEAPPROFD_HEAP_NAME_SZ];
  void (*enabled_callback)(void*, const AHeapProfileEnableCallbackInfo*);
  void (*disabled_callback)(void*, const AHeapProfileDisableCallbackInfo*);
  void* enabled_callback_data;
  void* disabled_callback_data;

  // Internal fields.
  perfetto::profiling::Sampler sampler;
  std::atomic<bool> ready;
  std::atomic<bool> enabled;
  std::atomic<uint64_t> adaptive_sampling_shmem_threshold;
  std::atomic<uint64_t> adaptive_sampling_max_sampling_interval_bytes;
};

struct AHeapProfileEnableCallbackInfo {
  uint64_t sampling_interval;
};

struct AHeapProfileDisableCallbackInfo {};

namespace {

using perfetto::profiling::ScopedSpinlock;
using perfetto::profiling::UnhookedAllocator;

#if defined(__GLIBC__)
const char* getprogname() {
  return program_invocation_short_name;
}
#elif !defined(__BIONIC__)
const char* getprogname() {
  return "";
}
#endif

// Holds the active profiling client. Is empty at the start, or after we've
// started shutting down a profiling session. Hook invocations take shared_ptr
// copies (ensuring that the client stays alive until no longer needed), and do
// nothing if this primary pointer is empty.
//
// This shared_ptr itself is protected by g_client_lock. Note that shared_ptr
// handles are not thread-safe by themselves:
// https://en.cppreference.com/w/cpp/memory/shared_ptr/atomic
//
// To avoid on-destruction re-entrancy issues, this shared_ptr needs to be
// constructed with an allocator that uses the unhooked malloc & free functions.
// See UnhookedAllocator.
//
// We initialize this storage the first time GetClientLocked is called. We
// cannot use a static initializer because that leads to ordering problems
// of the ELF's constructors.

alignas(perfetto::profiling::Client) char g_client_arr[sizeof(
    perfetto::profiling::Client)];

std::atomic<bool> g_client_active;
constexpr int64_t kClientShutdown = std::numeric_limits<int64_t>::min();
std::atomic<int64_t> g_client_refcnt{kClientShutdown};

void DisableAllHeaps();

void DecrementRefCnt() {
  auto prev = g_client_refcnt.fetch_sub(1, std::memory_order_acq_rel);
  if (prev == 0) {
    int64_t expected = 0;
    if (g_client_refcnt.compare_exchange_strong(expected, kClientShutdown,
                                                std::memory_order_acq_rel)) {
      reinterpret_cast<perfetto::profiling::Client*>(g_client_arr)->~Client();
    }
  }
}

template <typename F>
bool WithClient(F fn) {
  if (!g_client_active.load(std::memory_order_acquire)) {
    return false;
  }
  auto prev = g_client_refcnt.fetch_add(1, std::memory_order_acq_rel);
  if (prev < 0) {
    return false;
  }
  bool r = fn(reinterpret_cast<perfetto::profiling::Client*>(g_client_arr));
  DecrementRefCnt();
  return r;
}

void ShutdownClient() {
  if (g_client_active.exchange(false, std::memory_order_release)) {
    DisableAllHeaps();
    DecrementRefCnt();
  }
}

void ClientReady() {
  g_client_refcnt.store(1, std::memory_order_release);
  g_client_active.store(true, std::memory_order_release);
}

constexpr auto kMinHeapId = 1;

AHeapInfo g_heaps[256];

AHeapInfo& GetHeap(uint32_t id) {
  return g_heaps[id];
}

// Protects g_client, and serves as an external lock for sampling decisions (see
// perfetto::profiling::Sampler).
//
// We rely on this atomic's destuction being a nop, as it is possible for the
// hooks to attempt to acquire the spinlock after its destructor should have run
// (technically a use-after-destruct scenario).
static_assert(std::is_trivially_destructible<std::atomic<bool>>::value,
              "lock must be trivially destructible.");
std::atomic<bool> g_client_lock{false};

std::atomic<uint32_t> g_next_heap_id{kMinHeapId};

// Called only if |g_client_lock| acquisition fails, which shouldn't happen
// unless we're in a completely unexpected state (which we won't know how to
// recover from). Tries to abort (SIGABRT) the whole process to serve as an
// explicit indication of a bug.
//
// Doesn't use PERFETTO_FATAL as that is a single attempt to self-signal (in
// practice - SIGTRAP), while abort() tries to make sure the process has
// exited one way or another.
__attribute__((noreturn, noinline)) void AbortOnSpinlockTimeout() {
  PERFETTO_ELOG(
      "Timed out on the spinlock - something is horribly wrong. "
      "Aborting whole process.");
  abort();
}

void DisableAllHeaps() {
  for (uint32_t i = kMinHeapId; i < g_next_heap_id.load(); ++i) {
    AHeapInfo& info = GetHeap(i);
    if (!info.ready.load(std::memory_order_acquire))
      continue;
    if (info.enabled.load(std::memory_order_acquire)) {
      info.enabled.store(false, std::memory_order_release);
      if (info.disabled_callback) {
        AHeapProfileDisableCallbackInfo disable_info;
        info.disabled_callback(info.disabled_callback_data, &disable_info);
      }
    }
  }
}

uint64_t MaybeToggleHeap(uint32_t heap_id,
                         perfetto::profiling::Client* client) {
  AHeapInfo& heap = GetHeap(heap_id);
  if (!heap.ready.load(std::memory_order_acquire))
    return 0;
  auto interval =
      GetHeapSamplingInterval(client->client_config(), heap.heap_name);
  // The callbacks must be called while NOT LOCKED. Because they run
  // arbitrary code, it would be very easy to build a deadlock.
  if (interval) {
    AHeapProfileEnableCallbackInfo session_info{interval};
    if (!heap.enabled.load(std::memory_order_acquire) &&
        heap.enabled_callback) {
      heap.enabled_callback(heap.enabled_callback_data, &session_info);
    }
    heap.adaptive_sampling_shmem_threshold.store(
        client->client_config().adaptive_sampling_shmem_threshold,
        std::memory_order_relaxed);
    heap.adaptive_sampling_max_sampling_interval_bytes.store(
        client->client_config().adaptive_sampling_max_sampling_interval_bytes,
        std::memory_order_relaxed);
    heap.enabled.store(true, std::memory_order_release);
    client->RecordHeapInfo(heap_id, &heap.heap_name[0], interval);
  } else if (heap.enabled.load(std::memory_order_acquire)) {
    heap.enabled.store(false, std::memory_order_release);
    if (heap.disabled_callback) {
      AHeapProfileDisableCallbackInfo info;
      heap.disabled_callback(heap.disabled_callback_data, &info);
    }
  }
  return interval;
}

// We're a library loaded into a potentially-multithreaded process, which might
// not be explicitly aware of this possiblity. Deadling with forks/clones is
// extremely complicated in such situations, but we attempt to handle certain
// cases.
//
// There are two classes of forking processes to consider:
//  * well-behaved processes that fork only when their threads (if any) are at a
//    safe point, and therefore not in the middle of our hooks/client.
//  * processes that fork with other threads in an arbitrary state. Though
//    technically buggy, such processes exist in practice.
//
// This atfork handler follows a crude lowest-common-denominator approach, where
// to handle the latter class of processes, we systematically leak any |Client|
// state (present only when actively profiling at the time of fork) in the
// postfork-child path.
//
// The alternative with acquiring all relevant locks in the prefork handler, and
// releasing the state postfork handlers, poses a separate class of edge cases,
// and is not deemed to be better as a result.
//
// Notes:
// * this atfork handler fires only for the |fork| libc entrypoint, *not*
//   |clone|. See client.cc's |IsPostFork| for some best-effort detection
//   mechanisms for clone/vfork.
// * it should be possible to start a new profiling session in this child
//   process, modulo the bionic's heapprofd-loading state machine being in the
//   right state.
// * we cannot avoid leaks in all cases anyway (e.g. during shutdown sequence,
//   when only individual straggler threads hold onto the Client).
void AtForkChild() {
  PERFETTO_LOG("heapprofd_client: handling atfork.");

  // A thread (that has now disappeared across the fork) could have been holding
  // the spinlock. We're now the only thread post-fork, so we can reset the
  // spinlock, though the state it protects (the |g_client| shared_ptr) might
  // not be in a consistent state.
  g_client_lock.store(false);

  DisableAllHeaps();

  // Leak the existing client contents, including the profiling |Client| if
  // profiling was active at the time of the fork.
  g_client_refcnt.store(kClientShutdown, std::memory_order_release);
  g_client_active.store(false, std::memory_order_release);
}

}  // namespace

__attribute__((visibility("default"))) uint64_t
AHeapProfileEnableCallbackInfo_getSamplingInterval(
    const AHeapProfileEnableCallbackInfo* session_info) {
  return session_info->sampling_interval;
}

__attribute__((visibility("default"))) AHeapInfo* AHeapInfo_create(
    const char* heap_name) {
  size_t len = strlen(heap_name);
  if (len >= sizeof(AHeapInfo::heap_name)) {
    return nullptr;
  }

  uint32_t next_id = g_next_heap_id.fetch_add(1);
  if (next_id >= perfetto::base::ArraySize(g_heaps)) {
    return nullptr;
  }

  if (next_id == kMinHeapId)
    perfetto::profiling::StartHeapprofdIfStatic();

  AHeapInfo& info = GetHeap(next_id);
  strncpy(info.heap_name, heap_name, sizeof(info.heap_name));
  return &info;
}

__attribute__((visibility("default"))) AHeapInfo* AHeapInfo_setEnabledCallback(
    AHeapInfo* info,
    void (*callback)(void*, const AHeapProfileEnableCallbackInfo*),
    void* data) {
  if (info == nullptr)
    return nullptr;
  if (info->ready.load(std::memory_order_relaxed)) {
    PERFETTO_ELOG(
        "AHeapInfo_setEnabledCallback called after heap was registered. "
        "This is always a bug.");
    return nullptr;
  }
  info->enabled_callback = callback;
  info->enabled_callback_data = data;
  return info;
}

__attribute__((visibility("default"))) AHeapInfo* AHeapInfo_setDisabledCallback(
    AHeapInfo* info,
    void (*callback)(void*, const AHeapProfileDisableCallbackInfo*),
    void* data) {
  if (info == nullptr)
    return nullptr;
  if (info->ready.load(std::memory_order_relaxed)) {
    PERFETTO_ELOG(
        "AHeapInfo_setDisabledCallback called after heap was registered. "
        "This is always a bug.");
    return nullptr;
  }
  info->disabled_callback = callback;
  info->disabled_callback_data = data;
  return info;
}

__attribute__((visibility("default"))) uint32_t AHeapProfile_registerHeap(
    AHeapInfo* info) {
  if (info == nullptr)
    return 0;
  info->ready.store(true, std::memory_order_release);
  auto heap_id = static_cast<uint32_t>(info - &g_heaps[0]);

  // Enable the heap immediately if there's a matching ongoing session.
  WithClient([heap_id, info](perfetto::profiling::Client* client) {
    uint64_t interval = MaybeToggleHeap(heap_id, client);
    if (interval) {
      ScopedSpinlock s(&g_client_lock, ScopedSpinlock::Mode::Try);
      if (PERFETTO_UNLIKELY(!s.locked()))
        AbortOnSpinlockTimeout();
      info->sampler.SetSamplingInterval(interval);
    }
    return true;
  });
  return heap_id;
}

__attribute__((visibility("default"))) bool
AHeapProfile_reportAllocation(uint32_t heap_id, uint64_t id, uint64_t size) {
  AHeapInfo& heap = GetHeap(heap_id);
  if (!heap.enabled.load(std::memory_order_acquire)) {
    return false;
  }
  return WithClient([&heap, heap_id, id,
                     size](perfetto::profiling::Client* client) {
    size_t sampled_alloc_sz = 0;
    {
      ScopedSpinlock s(&g_client_lock, ScopedSpinlock::Mode::Try);
      if (PERFETTO_UNLIKELY(!s.locked()))
        AbortOnSpinlockTimeout();

      if (s.blocked_us()) {
        client->AddClientSpinlockBlockedUs(s.blocked_us());
      }

      sampled_alloc_sz = heap.sampler.SampleSize(static_cast<size_t>(size));
      if (sampled_alloc_sz == 0)  // not sampling
        return false;
      if (client->write_avail() < client->adaptive_sampling_shmem_threshold()) {
        bool should_increment = true;
        if (client->adaptive_sampling_max_sampling_interval_bytes() != 0) {
          should_increment =
              heap.sampler.sampling_interval() <
              client->adaptive_sampling_max_sampling_interval_bytes();
        }
        if (should_increment) {
          uint64_t new_interval = 2 * heap.sampler.sampling_interval();
          heap.sampler.SetSamplingInterval(2 *
                                           heap.sampler.sampling_interval());
          client->RecordHeapInfo(heap_id, "", new_interval);
        }
      }
    }  // unlock

    if (!client->RecordMalloc(heap_id, sampled_alloc_sz, size, id)) {
      ShutdownClient();
      return false;
    }
    return true;
  });
}

__attribute__((visibility("default"))) bool
AHeapProfile_reportSample(uint32_t heap_id, uint64_t id, uint64_t size) {
  const AHeapInfo& heap = GetHeap(heap_id);
  if (!heap.enabled.load(std::memory_order_acquire)) {
    return false;
  }
  return WithClient([heap_id, size, id](perfetto::profiling::Client* client) {
    if (!client->RecordMalloc(heap_id, size, size, id)) {
      ShutdownClient();
      return false;
    }
    return true;
  });
}

__attribute__((visibility("default"))) void AHeapProfile_reportFree(
    uint32_t heap_id,
    uint64_t id) {
  const AHeapInfo& heap = GetHeap(heap_id);
  if (!heap.enabled.load(std::memory_order_acquire)) {
    return;
  }
  WithClient([heap_id, id](perfetto::profiling::Client* client) {
    if (!client->RecordFree(heap_id, id))
      ShutdownClient();
    return true;
  });
}

__attribute__((visibility("default"))) bool AHeapProfile_initSession(
    void* (*malloc_fn)(size_t),
    void (*free_fn)(void*)) {
  static bool first_init = true;
  // Install an atfork handler to deal with *some* cases of the host forking.
  // The handler will be unpatched automatically if we're dlclosed.
  if (first_init && pthread_atfork(/*prepare=*/nullptr, /*parent=*/nullptr,
                                   &AtForkChild) != 0) {
    PERFETTO_PLOG("%s: pthread_atfork failed, not installing hooks.",
                  getprogname());
    return false;
  }
  first_init = false;

  bool connected = false;
  WithClient([&connected](perfetto::profiling::Client* client) {
    connected = client->IsConnected();
    return true;
  });
  if (connected) {
    PERFETTO_LOG("%s: Rejecting concurrent profiling initialization.",
                 getprogname());
    return true;  // success as we're in a valid state
  }
  ShutdownClient();

  int64_t refcnt = 1;  // fix incorrect "could be uninitialized" below.
  for (size_t i = 1; i < 10 * 1024; ++i) {
    refcnt = g_client_refcnt.load(std::memory_order_acquire);
    if (refcnt < 0)
      break;
    if (i % 1024 == 0)
      usleep(1000);
  }

  if (refcnt > 0) {  // Client still alive
    PERFETTO_LOG("%s: client in use on initialization.", getprogname());
    return false;
  }

  // The dispatch table never changes, so let the custom allocator retain the
  // function pointers directly.
  UnhookedAllocator<perfetto::profiling::Client> unhooked_allocator(malloc_fn,
                                                                    free_fn);

  if (!perfetto::profiling::ConstructClient(&g_client_arr)) {
    PERFETTO_LOG("%s: heapprofd_client not initialized, not installing hooks.",
                 getprogname());
    return false;
  }

  uint64_t heap_intervals[perfetto::base::ArraySize(g_heaps)] = {};
  uint32_t max_heap = g_next_heap_id.load();
  for (uint32_t i = kMinHeapId; i < max_heap; ++i) {
    heap_intervals[i] = MaybeToggleHeap(
        i, reinterpret_cast<perfetto::profiling::Client*>(g_client_arr));
  }

  PERFETTO_LOG("%s: heapprofd_client initialized.", getprogname());
  {
    ScopedSpinlock s(&g_client_lock, ScopedSpinlock::Mode::Try);
    if (PERFETTO_UNLIKELY(!s.locked()))
      AbortOnSpinlockTimeout();

    // This needs to happen under the lock for mutual exclusion regarding the
    // random engine.
    for (uint32_t i = kMinHeapId; i < max_heap; ++i) {
      AHeapInfo& heap = GetHeap(i);
      if (heap_intervals[i]) {
        heap.sampler.SetSamplingInterval(heap_intervals[i]);
      }
    }
  }
  ClientReady();
  return true;
}
