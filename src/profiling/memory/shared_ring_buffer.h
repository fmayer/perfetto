/*
 * Copyright (C) 2019 The Android Open Source Project
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

#ifndef SRC_PROFILING_MEMORY_SHARED_RING_BUFFER_H_
#define SRC_PROFILING_MEMORY_SHARED_RING_BUFFER_H_

#include "perfetto/ext/base/optional.h"
#include "perfetto/ext/base/unix_socket.h"
#include "perfetto/ext/base/utils.h"

#include <atomic>
#include <map>
#include <memory>

#include <stdint.h>

namespace perfetto {
namespace profiling {

// A concurrent, multi-writer single-reader ring buffer FIFO, based on a
// circular buffer over shared memory. It has similar semantics to a SEQ_PACKET
// + O_NONBLOCK socket, specifically:
//
// - Writes are atomic, data is either written fully in the buffer or not.
// - New writes are discarded if the buffer is full.
// - If a write succeeds, the reader is guaranteed to see the whole buffer.
// - Reads are atomic, no fragmentation.
// - The reader sees writes in write order (% discarding).
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// *IMPORTANT*: The ring buffer must be written under the assumption that the
// other end modifies arbitrary shared memory without holding the spin-lock.
// This means we must make local copies of read and write pointers for doing
// bounds checks followed by reads / writes, as they might change in the
// meantime.
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// TODO:
// - Write a benchmark.
class SharedRingBuffer {
 public:
  class Buffer {
   public:
    Buffer() {}
    Buffer(uint8_t* d, size_t s, uint64_t f)
        : data(d), size(s), bytes_free(f) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    explicit operator bool() const { return data != nullptr; }

    uint8_t* data = nullptr;
    size_t size = 0;
    uint64_t bytes_free = 0;
  };

  struct Stats {
    uint64_t bytes_written;
    uint64_t num_writes_succeeded;
    uint64_t num_writes_corrupt;
    uint64_t num_writes_overflow;

    uint64_t num_reads_succeeded;
    uint64_t num_reads_corrupt;
    uint64_t num_reads_nodata;

    uint64_t client_spinlock_blocked_us;
    bool hit_timeout;
  };

  static base::Optional<SharedRingBuffer> Create(size_t);
  static base::Optional<SharedRingBuffer> Attach(base::ScopedFile);

  ~SharedRingBuffer();
  SharedRingBuffer() = default;

  SharedRingBuffer(SharedRingBuffer&&) noexcept;
  SharedRingBuffer& operator=(SharedRingBuffer&&) noexcept;

  bool is_valid() const { return !!mem_; }
  size_t size() const { return size_; }
  int fd() const { return *mem_fd_; }
  size_t write_avail() {
    auto pos = GetPointerPositions();
    if (!pos)
      return 0;
    return write_avail(*pos);
  }

  Buffer BeginWrite(size_t size);
  void EndWrite(Buffer buf);

  Buffer BeginRead();
  void EndRead(Buffer);

  Stats GetStats() {
    Stats stats;

    stats.bytes_written = meta_->bytes_written.load(std::memory_order_relaxed);
    stats.num_writes_succeeded =
        meta_->num_writes_succeeded.load(std::memory_order_relaxed);
    stats.num_writes_corrupt =
        meta_->num_writes_corrupt.load(std::memory_order_relaxed);
    stats.num_writes_overflow =
        meta_->num_writes_overflow.load(std::memory_order_relaxed);

    stats.num_reads_succeeded =
        meta_->num_reads_succeeded.load(std::memory_order_relaxed);
    stats.num_reads_corrupt =
        meta_->num_reads_corrupt.load(std::memory_order_relaxed);
    stats.num_reads_nodata =
        meta_->num_reads_nodata.load(std::memory_order_relaxed);

    stats.client_spinlock_blocked_us =
        meta_->client_spinlock_blocked_us.load(std::memory_order_relaxed);
    stats.hit_timeout = meta_->hit_timeout.load(std::memory_order_relaxed);

    return stats;
  }

  void SetHitTimeout() { meta_->hit_timeout.store(true); }

  void AddClientSpinlockBlockedUs(size_t n) {
    meta_->client_spinlock_blocked_us.fetch_add(n, std::memory_order_relaxed);
  }

  uint64_t client_spinlock_blocked_us() {
    return meta_->client_spinlock_blocked_us.load(std::memory_order_relaxed);
  }

  void SetShuttingDown() {
    meta_->shutting_down.store(true, std::memory_order_relaxed);
  }

  bool shutting_down() {
    return meta_->shutting_down.load(std::memory_order_relaxed);
  }

  void SetReaderPaused() {
    meta_->reader_paused.store(true, std::memory_order_relaxed);
  }

  bool GetAndResetReaderPaused() {
    return meta_->reader_paused.exchange(false, std::memory_order_relaxed);
  }

  // Exposed for fuzzers.
  struct MetadataPage {
    alignas(uint64_t) std::atomic<bool> spinlock;
    std::atomic<uint64_t> read_pos;
    std::atomic<uint64_t> write_pos;

    std::atomic<uint64_t> failed_spinlocks;
    alignas(uint64_t) std::atomic<bool> shutting_down;
    alignas(uint64_t) std::atomic<bool> reader_paused;

    std::atomic<uint64_t> bytes_written;
    std::atomic<uint64_t> num_writes_succeeded;
    std::atomic<uint64_t> num_writes_corrupt;
    std::atomic<uint64_t> num_writes_overflow;

    std::atomic<uint64_t> num_reads_succeeded;
    std::atomic<uint64_t> num_reads_corrupt;
    std::atomic<uint64_t> num_reads_nodata;

    std::atomic<uint64_t> client_spinlock_blocked_us;
    alignas(uint64_t) std::atomic<bool> hit_timeout;
  };

 private:
  struct PointerPositions {
    uint64_t read_pos;
    uint64_t write_pos;
  };

  struct CreateFlag {};
  struct AttachFlag {};
  SharedRingBuffer(const SharedRingBuffer&) = delete;
  SharedRingBuffer& operator=(const SharedRingBuffer&) = delete;
  SharedRingBuffer(CreateFlag, size_t size);
  SharedRingBuffer(AttachFlag, base::ScopedFile mem_fd) {
    // See comment in CreateFlag constructor for details on this check.
    if (!std::atomic<uint64_t>{}.is_lock_free()) {
      PERFETTO_ELOG("No lock-free uint64_t. Cannot use SharedRingBuffer.");
      return;
    }
    Initialize(std::move(mem_fd));
  }

  void Initialize(base::ScopedFile mem_fd);
  bool IsCorrupt(const PointerPositions& pos);

  inline base::Optional<PointerPositions> GetPointerPositions() {
    PointerPositions pos;
    // We need to acquire load the read_pos to make sure we observe a
    // consistent ring buffer in BeginRead, otherwise it is possible that we
    // observe the read_pos increment, but the memset(0) in the previous
    // EndWrite, thus reading a stale payload size.
    //
    // This is matched by a release at the end of EndRead.
    pos.read_pos = meta_->read_pos.load(std::memory_order_acquire);
    // Read the write pos afterwards to ensure write_pos >= read_pos.
    pos.write_pos = meta_->write_pos.load(std::memory_order_relaxed);

    base::Optional<PointerPositions> result;
    if (IsCorrupt(pos))
      return result;
    result = pos;
    return result;
  }

  inline size_t read_avail(const PointerPositions& pos) {
    PERFETTO_DCHECK(pos.write_pos >= pos.read_pos);
    auto res = static_cast<size_t>(pos.write_pos - pos.read_pos);
    PERFETTO_DCHECK(res <= size_);
    return res;
  }

  inline size_t write_avail(const PointerPositions& pos) {
    return size_ - read_avail(pos);
  }

  inline uint8_t* at(uint64_t pos) { return mem_ + (pos & (size_ - 1)); }

  base::ScopedFile mem_fd_;
  MetadataPage* meta_ = nullptr;  // Start of the mmaped region.
  uint8_t* mem_ = nullptr;  // Start of the contents (i.e. meta_ + kPageSize).

  // Size of the ring buffer contents, without including metadata or the 2nd
  // mmap.
  size_t size_ = 0;

  // Remember to update the move ctor when adding new fields.
};

}  // namespace profiling
}  // namespace perfetto

#endif  // SRC_PROFILING_MEMORY_SHARED_RING_BUFFER_H_
