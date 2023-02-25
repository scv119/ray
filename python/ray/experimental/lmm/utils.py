import time

import ray


@ray.remote(num_gpu=1)
class GpuActor:
    def __init__(self, id, model):
        self.id = id
        self.model = model
        pass

    def forward(self, input):
        # TODO
        return

    def backward(self, grad, value):
        # TODO
        pass


def _forward(layer_id, input):
    return ray.get_actor(layer_id).forward(input)


def _backward(layer_id, input):
    (grad, value) = input
    return ray.get_actor(layer_id).backward(grad, value)


class RayDataMode:
    def __init__(self, layers):
        for layer_id, layer in enumerate(layers):
            self.layer_actors = [
                GpuActor.options(name=str(layer_id)).remote(layer_id, layer)
            ]

    def train(self, ds):
        # forward pass
        for _ in range(len(self.layer_actors)):
            ds = ds.map(_forward)

        # loss functions?

        # backward pass
        for _ in reversed(range(len(self.layer_actors))):
            ds = ds.map(_backward)

        # ds = ds.map(...)

        # actual training
        for _ in ds.iter_batches():
            pass

    def inference(self):
        pass
