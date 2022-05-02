import uuid
import cupy as cp
import numpy as np
import ray
import ray.util.collective as collective
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

GROUP_NAME = "experimental_nccl_group_name"

# Represents a GPU object managed by Ray.
class GpuObjectRef:
    def __init__(self, id, src_rank, shape, dtype):
        self.id = id
        self.src_rank = src_rank
        self.shape = shape
        self.dtype = dtype
        # self.host_name = ...


# Per device GPU object manager.
class DeviceGpuObjectManagerBase:
    def __init__(self):
        self.buffers = {}

    def setup(self, world_size: int, rank: int, group_name):
        self.world_size = world_size
        self.rank = rank
        self.group_name = group_name

    def put_numpy_array(self, buffer: np.ndarray) -> GpuObjectRef:
        buffer_id = uuid.uuid4()
        self.buffers[buffer_id] = cp.asarray(buffer)
        return GpuObjectRef(buffer_id, self.rank, buffer.shape, buffer.dtype)

    def put_cupy_array(self, buffer: cp.ndarray) -> GpuObjectRef:
        buffer_id = uuid.uuid4()
        self.buffers[buffer_id] = buffer
        return GpuObjectRef(buffer_id, self.rank, buffer.shape, buffer.dtype)

    # TODO: support torch tensor
    # def put_torch_tensor()
    #   ...

    def get_gpu_buffer(self, ref: GpuObjectRef):
        assert self.contains(ref)
        return self.buffers[ref.id]

    def contains(self, ref: GpuObjectRef) -> bool:
        return ref.id in self.buffers

    def send_buffer(self, ref: GpuObjectRef, dest_rank: int) -> None:
        assert self.contains(ref)
        collective.send(self.buffers[ref.id], dest_rank, self.group_name)

    def receive_buffer(self, ref: GpuObjectRef, src_rank: int):
        assert not self.contains(ref)
        self.buffers[ref.id] = cp.ndarray(shape=ref.shape, dtype=ref.dtype)
        collective.recv(self.buffers[ref.id], src_rank, self.group_name)

    # TODO: support GC objects


# Per Host Gpu object manager.
@ray.remote(num_gpus=0, num_cpus=0)
class HostGpuObjectManager:
    def __init__(self, group_name=GROUP_NAME):
        self.group_name = group_name
        self.actors = []

    def register_gpu_actor(self, actor: "ActorHandle"):
        self.actors.append(actor)

    def setup_collective_group(self):
        ranks = list(range(len(self.actors)))
        _options = {
            "group_name": GROUP_NAME,
            "world_size": len(self.actors),
            "ranks": ranks,
            "backend": "nccl",
        }
        collective.declare_collective_group(self.actors, **_options)
        ray.get(
            [
                actor.setup.remote(len(self.actors), rank, self.group_name)
                for rank, actor in enumerate(self.actors)
            ]
        )

    def transfer_gpu_object(self, ref: GpuObjectRef, src: int, dst: int):
        send = self.gpu_managers[src].send_buffer.remote(ref, dst)
        receive = self.gpu_managers[dst].receive_buffer.remote(ref, src)
        ray.get([send, receive])


def get_or_create_host_gpu_object_manager() -> "ActorHandle":
    node_id = ray.get_runtime_context().node_id
    actor_name = f"host-gpu-object-manager-{node_id}"
    return HostGpuObjectManager.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        name=actor_name,
        namespace="GPU_TEST",
        get_if_exists=True,
    ).remote()


# examples on how to use it:

if __name__ == "__main__":

    @ray.remote(num_gpus=1)
    class SenderActor(DeviceGpuObjectManagerBase):
        def create_and_send(self, receiver):
            object = cp.ones((4,), dtype=cp.float32)
            ref = self.put_cupy_array(object)
            return receiver.receive_gpu_ref.remote(ref)

    @ray.remote(num_gpus=1)
    class ReceiverActor(DeviceGpuObjectManagerBase):
        def receive_gpu_ref(self, tensor_ref: GpuObjectRef):
            buffer = self.get_gpu_buffer(tensor_ref)
            print(buffer)

    sender_actor = SenderActor.remote()
    receiver_actor = ReceiverActor.remote()

    host_gpu_object_manager = get_or_create_host_gpu_object_manager()
    host_gpu_object_manager.register_gpu_actor(sender_actor)
    host_gpu_object_manager.register_gpu_actor(receiver_actor)

    # setup collective group
    ray.get(host_gpu_object_manager.setup_collective_group.remote())

    # do the actuall function call.
    ray.get(sender_actor.create_and_send(receiver_actor))
