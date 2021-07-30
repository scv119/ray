// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/object_manager/plasma/object_store.h"
#include <limits>
#include "absl/random/random.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"

using namespace ray;

// namespace plasma {
// namespace {
// template <typename T>
// T Random(T max = std::numeric_limits<T>::max()) {
//   static absl::BitGen bitgen;
//   return absl::Uniform(bitgen, 0, max);
// }

// Allocation CreateAllocation(int64_t size) {
//   return Allocation(
//       /* address */ nullptr, size,
//       /* fd */ {Random<int>(), Random<int64_t>()},
//       /* offset */ Random<ptrdiff_t>(),
//       /* device_num */ 0,
//       /* mmap_size */ Random<int64_t>());
// }

// const std::string Serialize(const Allocation &allocation) {
//   return absl::StrFormat("%p/%d/%d/%d/%d/%d/%d", allocation.address, allocation.size,
//                          allocation.fd.first, allocation.fd.second, allocation.offset,
//                          allocation.device_num, allocation.mmap_size);
// }

// ObjectInfo CreateObjectInfo(ObjectID object_id, int64_t object_size) {
//   ObjectInfo info;
//   info.object_id = object_id;
//   info.data_size = Random<int64_t>(object_size);
//   info.metadata_size = object_size - info.data_size;
//   info.owner_raylet_id = NodeID::FromRandom();
//   info.owner_ip_address = "random_ip";
//   info.owner_port = Random<int>();
//   info.owner_worker_id = WorkerID::FromRandom();
//   return info;
// }
// }  // namespace

// TEST(ObjectStoreTest, PassThroughTest) {
//   ObjectStore store;
//   auto id = ObjectID::FromRandom();
//   auto id1 = ObjectID::FromRandom();
//   while (id1 == id) {
//     id1 = ObjectID::FromRandom();
//   }

//   {
//     auto allocation = CreateAllocation(10);
//     auto alloc_str = Serialize(allocation);
//     auto info = CreateObjectInfo(id, 10);
//     auto entry = store.CreateObject(std::move(allocation), info, {});
//     EXPECT_NE(entry, nullptr);
//     EXPECT_EQ(entry->ref_count, 0);
//     EXPECT_EQ(entry->state, ObjectState::PLASMA_CREATED);
//     EXPECT_EQ(alloc_str, Serialize(entry->allocation));
//     EXPECT_EQ(info, entry->object_info);
//     EXPECT_EQ(store.GetNumBytesCreatedTotal(), 10);
//     EXPECT_EQ(store.GetNumBytesUnsealed(), 10);
//     EXPECT_EQ(store.GetNumObjectsUnsealed(), 1);

//     // verify get
//     auto entry1 = store.GetObject(id);
//     EXPECT_EQ(entry1, entry);

//     // get non exists
//     auto entry2 = store.GetObject(id1);
//     EXPECT_EQ(entry2, nullptr);

//     // seal object
//     auto entry3 = store.SealObject(id);
//     EXPECT_EQ(entry3, entry);
//     EXPECT_EQ(entry3->state, ObjectState::PLASMA_SEALED);
//     EXPECT_EQ(store.GetNumBytesCreatedTotal(), 10);
//     EXPECT_EQ(store.GetNumBytesUnsealed(), 0);
//     EXPECT_EQ(store.GetNumObjectsUnsealed(), 0);

//     // delete sealed
//     auto allocation1 = store.DeleteObject(id);
//     EXPECT_EQ(alloc_str, Serialize(allocation1));
//     EXPECT_EQ(nullptr, store.GetObject(id));
//   }

//   {
//     auto allocation = CreateAllocation(12);
//     auto alloc_str = Serialize(allocation);
//     auto info = CreateObjectInfo(id1, 12);
//     auto entry = store.CreateObject(std::move(allocation), info, {});
//     EXPECT_NE(entry, nullptr);
//     EXPECT_EQ(entry->ref_count, 0);
//     EXPECT_EQ(entry->state, ObjectState::PLASMA_CREATED);
//     EXPECT_EQ(alloc_str, Serialize(entry->allocation));
//     EXPECT_EQ(info, entry->object_info);
//     EXPECT_EQ(store.GetNumBytesCreatedTotal(), 22);
//     EXPECT_EQ(store.GetNumBytesUnsealed(), 12);
//     EXPECT_EQ(store.GetNumObjectsUnsealed(), 1);

//     // delete unsealed
//     auto allocation1 = store.DeleteObject(id1);
//     EXPECT_EQ(alloc_str, Serialize(allocation1));
//     EXPECT_EQ(nullptr, store.GetObject(id1));

//     EXPECT_EQ(store.GetNumBytesCreatedTotal(), 22);
//     EXPECT_EQ(store.GetNumBytesUnsealed(), 0);
//     EXPECT_EQ(store.GetNumObjectsUnsealed(), 0);
//   }
// }
// }  // namespace plasma

// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
