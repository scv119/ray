import os

import ray
import torch

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
n, m = 128, 4

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
        tensor = torch.ones(1).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(tensor)

ray.init(address="local")

actors = [Actor.remote(rank=i, size=2) for i in range(2)]
ray.get([actor.run.remote() for actor in actors])
