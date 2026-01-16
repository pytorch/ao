#pragma once
#include <cstdint>

enum ProfilerTag {
  TMALoad = 0,
  ProcessThenTMAStore = 1,
};

__device__ inline int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct Profiler {
  int64_t *data_ptr_;
  int sm_id_;
  int cnt_;
  int max_entries_;

  // Pending event tracking: map (tag, chunk_idx) -> slot index
  // We use a simple array indexed by: tag * MAX_CHUNKS + chunk_idx
  static constexpr int MAX_CHUNKS = 32;  // Should be >= CHUNKS_PER_TB
  int pending_slots_[2 * MAX_CHUNKS];  // -1 means no pending event

  __device__ void init(int num_entries, int64_t *data_ptr, int bid) {
    data_ptr_ = data_ptr + bid * (1 + num_entries * 5);
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(sm_id_));
    cnt_ = 0;
    max_entries_ = num_entries;

    // Initialize pending slots to -1 (no pending events)
    for (int i = 0; i < 2 * MAX_CHUNKS; i++) {
      pending_slots_[i] = -1;
    }
  }

  __device__ void start(ProfilerTag tag, int chunk_idx) {
    if (cnt_ >= max_entries_) return;  // Bounds check

    // Allocate a new slot for this event
    int slot_idx = cnt_;

    // Store pending slot for this (tag, chunk_idx) pair
    int key = tag * MAX_CHUNKS + chunk_idx;
    pending_slots_[key] = slot_idx;

    // Write event metadata and start time to the slot
    data_ptr_[1 + slot_idx * 5 + 0] = sm_id_;
    data_ptr_[1 + slot_idx * 5 + 1] = tag;
    data_ptr_[1 + slot_idx * 5 + 2] = chunk_idx;
    data_ptr_[1 + slot_idx * 5 + 3] = globaltimer();
    data_ptr_[1 + slot_idx * 5 + 4] = 0;  // Duration placeholder

    cnt_ += 1;  // Increment counter immediately to reserve the slot
  }

  __device__ void stop(ProfilerTag tag, int chunk_idx) {
    // Find the pending slot for this (tag, chunk_idx) pair
    int key = tag * MAX_CHUNKS + chunk_idx;
    int slot_idx = pending_slots_[key];

    if (slot_idx < 0) return;  // No matching start event found

    // Calculate duration and write to the slot
    int64_t end_time = globaltimer();
    int64_t start_time = data_ptr_[1 + slot_idx * 5 + 3];
    data_ptr_[1 + slot_idx * 5 + 4] = end_time - start_time;

    // Clear the pending slot
    pending_slots_[key] = -1;
  }

  __device__ void flush() { data_ptr_[0] = cnt_; }
};
