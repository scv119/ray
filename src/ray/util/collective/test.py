import os

import ray
import torch
import ray_reducer

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
n, m = 128, 4


def all_reduce_tensors(tensors):
  print(tensors)


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, rank, size):
        del os.environ['CUDA_VISIBLE_DEVICES'] 
        import ray_collectives
        import torch.distributed as dist
        self.rank = rank
        dist.init_process_group("ray", rank=rank, world_size=size)
        print(f"my rank is {rank}")
        torch.cuda.set_device(rank)
        
    def run(self):
        import torch.distributed as dist
        print(f"[worker] my task_id {ray.get_runtime_context().get_task_id()}")
        tensor = torch.ones(5).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(tensor)

ray.init(address="local")
ray_reducer.set_up_ray_reduce([(ray.get_runtime_context().get_node_id(), 4)])

actors = [Actor.remote(rank=i, size=4) for i in range(4)]
ray.get([actor.run.remote() for actor in actors])
