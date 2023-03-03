#!/usr/bin/env python3

import ray
from collections import defaultdict

num_gpus = 4

@ray.remote
class Reducer:
    def __init__(self):
        self.callbacks = defaultdict(list)
        self.results = {}
        self.inputs = defaultdict(list)

    def reduce(self, tensor, global_order, num_expected, client_name):
        #print(f'reduce call from {client_name} {global_order}')

        self.inputs[global_order].append(tensor)
        self.callbacks[client_name].append(global_order)
        # hand off this to a thread for reduction
        # once the thread has finished for a particular bucket, then we send it back
        # to the reducer client.

    def temporary_flush(self):
        for client_name, invocations in self.callbacks.items():
            print(client_name, invocations)
            # TODO file bug for when actor is not stored
            actor = ray.get_actor(client_name)
            ray.get(actor.callback.remote(invocations))

        self.callbacks.clear()
        self.inputs.clear()
        self.results.clear()

class MonotonicCounter:
    def __init__(self):
        self.value = 0
    def get_and_increment(self):
        rval = self.value
        self.value += 1
        return rval

@ray.remote
class ReducerClient:
    def __init__(self, name, reducer):
        self.reducer = reducer
        self.name = name
        self.counter = MonotonicCounter()

    # make into async actor.
    def allreduce(self, gpu_buffer):
        # assign some global order id
        # copy to CPU
        # copy to reducer
        # get result from reducer
        # copy to GPU (gpu_buffer)
        tensor = None
        ray.get(self.reducer.reduce.remote(
            tensor,
            self.counter.get_and_increment(),
            num_gpus,
            self.name
        ))

    def callback(self, invocations):
        print('callback', invocations)
        pass

@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, name, rank, reducer):
        self.reducer = reducer
        self.reducer_client = ReducerClient.options(name=f"reducer_{name}").remote(f"reducer_{name}", reducer)
        self.rank = rank

    def do_training(self):
        for epoch in range(10):
            # create tensor on GPU (assume we're running in SPMD)
            # call allreduce on tensor
            ray.get(self.reducer_client.allreduce.remote(None))


if __name__ == '__main__':
    reducer = Reducer.remote()
    namegen = lambda i: f"trainer_{i}"
    trainers = [Trainer.options(name=namegen(i)).remote(namegen(i), i, reducer) for i in range(num_gpus)]
    ray.get([t.do_training.remote() for t in trainers])
    ray.get(reducer.temporary_flush.remote())
