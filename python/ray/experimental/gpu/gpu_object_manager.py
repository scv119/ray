import base64
import uuid
import cupy as cp
import numpy as np
import ray
import ray.util.collective as collective
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from flask import Flask
import requests
import pickle
import urllib.parse

from gpu_object_ref import GpuObjectRef

GROUP_NAME = "experimental_nccl_group_name"

# Represents a GPU object managed by Ray.
#class GpuObjectRef:
#    def __init__(self, id, src_rank, shape, dtype):
#        self.id = id
#        self.src_rank = src_rank
#        self.shape = shape
#        self.dtype = dtype
#        # self.host_name = ...

@ray.remote(num_cpus=0)
def send_request(url):
    requests.get(url)

# Per Host Gpu object manager.
@ray.remote(num_gpus=0, num_cpus=0)
class HostGpuObjectManager:
    def __init__(self, group_name=GROUP_NAME):
        self.group_name = group_name
        self.actors = []
        self.actor_ports = []

    def register_gpu_actor(self, actor: "ActorHandle", port):
        self.actors.append(actor)
        self.actor_ports.append(port)

    def setup_collective_group(self):
        ranks = list(range(len(self.actors)))
        _options = {
            "group_name": GROUP_NAME,
            "world_size": len(self.actors),
            "ranks": ranks,
            "backend": "nccl",
        }
        print("setup collective groups")
        collective.create_collective_group(self.actors, **_options)
        print("done")

        print("setup remote info")
        ray.get(
            [
                actor.setup.remote(len(self.actors), rank, self.group_name)
                for rank, actor in enumerate(self.actors)
            ]
        )
        print("done")

    def transfer_gpu_object(self, ref: GpuObjectRef, src: int, dst: int):
        print("start transfer")
        serialized = urllib.parse.quote(base64.b64encode(pickle.dumps(ref)).decode("utf-8"), safe='')
        send_url = f"http://127.0.0.1:{self.actor_ports[src]}/send?dst={dst}&ref={serialized}"
        recv_url = f"http://127.0.0.1:{self.actor_ports[dst]}/recv?src={src}&ref={serialized}"
        ray.get([
            send_request.remote(send_url),
            send_request.remote(recv_url),
            ])
        print("transfer succeeded")


def get_or_create_host_gpu_object_manager() -> "ActorHandle":
    node_id = ray.get_runtime_context().node_id
    actor_name = f"host-gpu-object-manager-{node_id}"
    return HostGpuObjectManager.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
        name=actor_name,
        namespace="GPU_TEST",
        get_if_exists=True,
    ).remote()


class HttpCoordinator:
    def __init__(self, port, send_fn, recv_fn):
        self.port = port
        self.app = Flask(__name__)
        self.send_fn = send_fn
        self.recv_fn = recv_fn 

        @self.app.route("/send")
        def send():
            from flask import request 
            dst = request.args.get("dst")
            ref = request.args.get("ref")
            self.send_fn(pickle.loads(base64.b64decode(ref)), int(dst))
            return "OK"

        @self.app.route("/recv")
        def recv():
            from flask import request 
            src = request.args.get("src")
            ref = request.args.get("ref")
            self.recv_fn(pickle.loads(base64.b64decode(ref)), int(src))
            return "OK"

    def run(self):
        self.app.run(port=self.port)


# Per device GPU object manager.
class DeviceGpuObjectManagerBase:
    def __init__(self, port):
        self.buffers = {}
        self.device_manager = None
        self.coordinator = HttpCoordinator(port, self.send_buffer, self.recv_buffer)

    def setup(self, world_size: int, rank: int, group_name):
        self.world_size = world_size
        self.rank = rank
        self.group_name = group_name

    def run_coordinator(self):
        self.coordinator.run()

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

    def _get_gpu_buffer(self, ref: GpuObjectRef):
        assert self.contains(ref)
        return self.buffers[ref.id]

    def _get_device_object_manager(self) -> "ActorHandle":
        if not self.device_manager:
            self.device_manager = get_or_create_host_gpu_object_manager()
        return self.device_manager

    def get_gpu_buffer(self, ref: GpuObjectRef):
        if self.contains(ref):
            return self._get_gpu_buffer(ref)
        ray.get(
            self._get_device_object_manager().transfer_gpu_object.remote(
                ref, ref.src_rank, self.rank
            )
        )
        return self._get_gpu_buffer(ref)

    def contains(self, ref: GpuObjectRef) -> bool:
        return ref.id in self.buffers

    def send_buffer(self, ref: GpuObjectRef, dest_rank: int) -> None:
        assert self.contains(ref)
        print("collective.send")
        collective.send(self.buffers[ref.id], dest_rank, self.group_name)
        print("collective.send done")

    def recv_buffer(self, ref: GpuObjectRef, src_rank: int):
        assert not self.contains(ref)
        self.buffers[ref.id] = cp.ndarray(shape=ref.shape, dtype=ref.dtype)
        print("collective.receive")
        collective.recv(self.buffers[ref.id], src_rank, self.group_name)
        print("collective.receive done")

    # TODO: support GC objects


# examples on how to use it:
if __name__ == "__main__":

    @ray.remote(num_gpus=1)
    class SenderActor(DeviceGpuObjectManagerBase):
        def create_and_send(self, receiver):
            object = cp.ones((4,), dtype=cp.float32)
            ref = self.put_cupy_array(object)
            return ray.get(receiver.receive_gpu_ref.remote(ref))

    @ray.remote(num_gpus=1)
    class ReceiverActor(DeviceGpuObjectManagerBase):
        def receive_gpu_ref(self, tensor_ref: GpuObjectRef):
            buffer = self.get_gpu_buffer(tensor_ref)
            print("buffer received!")
            print(buffer)

    sender_actor = SenderActor.options(max_concurrency=2, num_gpus=1).remote(5000)
    receiver_actor = ReceiverActor.options(max_concurrency=2, num_gpus=1).remote(5001)
    
    sender_actor.run_coordinator.remote()
    receiver_actor.run_coordinator.remote()

    # setup the actors.
    host_gpu_object_manager = get_or_create_host_gpu_object_manager()
    ray.get([
        host_gpu_object_manager.register_gpu_actor.remote(sender_actor, 5000),
        host_gpu_object_manager.register_gpu_actor.remote(receiver_actor, 5001)
    ])

    # setup collective group.
    ray.get(host_gpu_object_manager.setup_collective_group.remote())

    # do the actuall function call.
    ray.get(sender_actor.create_and_send.remote(receiver_actor))