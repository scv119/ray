import cupy as cp
import numpy as np
import ray
import time

# from gpu_object_ref import GpuObjectRef
# from gpu_object_manager import GpuActorBase, setup_transfer_group
from mock import GpuObjectRef
from mock import GpuActorBase, setup_transfer_group

TENSOR_SIZE = 10000
NUM_RUNS = 10


@ray.remote(num_gpus=1)
class GpuActor(GpuActorBase):
    """Example class for gpu transfer."""

    def put_gpu_obj(self):
        object = cp.random.rand(TENSOR_SIZE, TENSOR_SIZE, dtype=cp.float32)
        return self.put_gpu_buffer(object)

    def load_gpu_obj(self, tensor_ref: GpuObjectRef):
        buffer = self.get_gpu_buffer(tensor_ref)
        print(f"buffer received! {buffer.shape}")


if __name__ == "__main__":
    sender_actor = GpuActor.options(num_gpus=1).remote()
    receiver_actor = GpuActor.options(num_gpus=1).remote()
    setup_transfer_group([sender_actor, receiver_actor])

    stats = []
    for _ in range(NUM_RUNS):
        start = time.time()
        ref = sender_actor.put_gpu_obj.remote()
        ray.get(receiver_actor.load_gpu_obj.remote(ref))
        time_diff_ms = (time.time() - start) * 1000
        stats.append(time_diff_ms)

    mean = round(np.mean(stats), 2)
    std = round(np.std(stats), 2)
    print(
        f"2D Tensor dim: {TENSOR_SIZE}, mean_ms: {mean}, std_ms: {std}, num_runs: {NUM_RUNS}"
    )
