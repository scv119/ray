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
#pragma once

#include <gtest/gtest_prod.h>

#include "ray/gcs/autoscaler/node_provider.h"
#include "ray/gcs/gcs_server/gcs_resource_manager.h"

namespace ray {
namespace autoscaler {

class Autoscaler {
 public:
  Autoscaler(INodeProvider &node_provider, gcs::GcsResourceManager &resource_manager);
  void RunOnce();

 private:
  void GetNodesToLaunch();

 private:
  INodeProvider &node_provider_;
  gcs::GcsResourceManager &resource_manager_;
};
}  // namespace autoscaler
}  // namespace ray
