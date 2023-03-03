#!/usr/bin/env python3

import ray
from collections import defaultdict
import asyncio

num_gpus = 4

@ray.remote
class Reducer:
    def __init__(self):
        self.results = {}
        self.inputs = defaultdict(list)
        self.consumed_count = defaultdict(int)

    async def reduce(self, tensor, global_order, num_expected, client_name, client_rank):
        import torch
        self.inputs[global_order].append(tensor)

        # The zeroth rank will wait for all inputs, then sum, then store the result.
        # The non-zero ranks will wait for the zeroth rank to store the result.
        # The last rank to access the result will delete the results and inputs.
        if client_rank == 0:
            while len(self.inputs[global_order]) < num_expected:
                await asyncio.sleep(0.1)
        
            tensors = self.inputs[global_order]
            
            sum_tensors = tensors[0]
            print('summing tensors: ', tensors)
            for t in tensors[1:]:
                sum_tensors = torch.add(sum_tensors, t)

            self.results[global_order] = sum_tensors
            self.consumed_count[global_order] += 1
            result = sum_tensors
        else:
            while self.consumed_count[global_order] == 0:
                await asyncio.sleep(0.1)
            result = self.results[global_order]
            self.consumed_count[global_order] += 1

            if self.consumed_count[global_order] == num_expected:
                del self.inputs[global_order]
                del self.consumed_count[global_order]
                del self.results[global_order]
                print(f'released global_order {global_order}')

        # we have all of them.
        return result

    def assert_no_leaks(self):
        assert not self.inputs, f"inputs has elements when it shouldn't {self.inputs}"
        assert not self.consumed_count, f"consumed_count has elements when it shouldn't {self.consumed_count}"
        assert not self.results, f"results has elements when it shouldn't {self.results}"

class MonotonicCounter:
    def __init__(self):
        self.value = 0
    def get_and_increment(self):
        rval = self.value
        self.value += 1
        return rval

@ray.remote
class ReducerClient:
    def __init__(self, name, reducer, rank):
        import os
        del os.environ['CUDA_VISIBLE_DEVICES']

        self.reducer = reducer
        self.name = name
        self.rank = rank

    # make into async actor.
    async def allreduce(self, gpu_buffer, global_order):
        # assign some global order id
        # copy to CPU
        # copy to reducer
        # get result from reducer
        # copy to GPU (gpu_buffer)
        

        cpu_tensor = gpu_buffer.to('cpu')

        print(f'rank {self.rank} order {global_order} shape {cpu_tensor.size()}')

        reduced = await self.reducer.reduce.remote(
            cpu_tensor,
            global_order,
            num_gpus,
            self.name,
            self.rank,
        )
        
        # TODO nonblocking? otherwise this consumes the event loop
        gpu_buffer.copy_(reduced)


@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, name, rank, reducer):
        self.reducer = reducer
        self.reducer_client = ReducerClient.options(name=f"reducer_{name}").remote(f"reducer_{name}", reducer, rank)
        self.rank = rank

    def do_training(self):
        if self.rank == 0:
            print('importing torch')
        import torch

        if self.rank == 0:
            print('creating CUDA tensors')

        input_tensors = [torch.ones(i+1).cuda() for i in range(10)]
        counter = MonotonicCounter()

        for epoch in range(10):
            futures = []
            for t in input_tensors:
                futures.append(self.reducer_client.allreduce.remote(t, counter.get_and_increment()))
            ray.get(futures)


if __name__ == '__main__':
    print('creating actors')
    reducer = Reducer.remote()
    namegen = lambda i: f"trainer_{i}"
    trainers = [Trainer.options(name=namegen(i)).remote(namegen(i), i, reducer) for i in range(num_gpus)]
    print('starting training')
    ray.get([t.do_training.remote() for t in trainers])
    ray.get(reducer.assert_no_leaks.remote())
