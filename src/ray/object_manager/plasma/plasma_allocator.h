// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstddef>
#include <cstdint>

#include "absl/container/flat_hash_map.h"

namespace plasma {

class PlasmaAllocator {
 public:
  struct AllocationInfo;

  /// Allocates size bytes and returns a pointer to the allocated memory. The
  /// memory address will be a multiple of alignment, which must be a power of two.
  ///
  /// \param alignment Memory alignment.
  /// \param bytes Number of bytes.
  /// \return Pointer to allocated memory.
  static void *Memalign(size_t alignment, size_t bytes);

  // Same as MemAlign, but allocates pages from the filesystem. The footprint limit
  // is not enforced for these allocations, but allocations here are still tracked
  // and count towards the limit.
  static void *DiskMemalignUnlimited(size_t alignment, size_t bytes);

  /// Frees the memory space pointed to by mem, which must have been returned by
  /// a previous call to Memalign()
  ///
  /// \param mem Pointer to memory to free.
  /// \param bytes Number of bytes to be freed.
  static void Free(void *mem, size_t bytes);

  /// Sets the memory footprint limit for Plasma.
  ///
  /// \param bytes Plasma memory footprint limit in bytes.
  static void SetFootprintLimit(size_t bytes);

  /// Get the memory footprint limit for Plasma.
  ///
  /// \return Plasma memory footprint limit in bytes.
  static int64_t GetFootprintLimit();

  /// Get the number of bytes allocated by Plasma so far.
  /// \return Number of bytes allocated by Plasma so far.
  static int64_t Allocated();

  /// Get the number of bytes fallback allocated by Plasma so far.
  /// \return Number of bytes fallback allocated by Plasma so far.
  static int64_t FallbackAllocated();

  /// Get the AllocationInfo of an allocated address returned by
  /// Memalign/DiskMemalignUnlimited call. Returns an invaid
  /// AllocationInfo if no such allocation found
  static const AllocationInfo &GetAllocationInfo(void *address);

 public:
  struct AllocationInfo {
    /// The file descriptor of the memory mapped file where the memory allocated.
    MEMFD_TYPE fd;
    /// The offset in bytes in the memory mapped file of the allocated memory.
    ptrdiff_t offset;
    /// Device number of the allocated memory.
    int device_num;
    /// the total size of this mapped memory.
    int64_t mmap_size;
  };

 private:
  static int64_t allocated_;
  static int64_t fallback_allocated_;
  static int64_t footprint_limit_;
  static absl::flat_hash_map<void *, AllocationInfo> cached_allocation_info_;
};

}  // namespace plasma
