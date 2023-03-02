#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import torch.distributed
from ray.util.collective.naive_collective_group import NaiveProcessGroup


def create_ray_process_group(store, rank, size, timeout):
    return NaiveProcessGroup(
        store,
        rank,
        size,
        NaiveProcessGroup.Options(
            timeout=timeout,
        ),
    )


torch.distributed.Backend.register_backend("ray_collective", NaiveProcessGroup)
