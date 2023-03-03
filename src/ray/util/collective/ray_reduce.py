import ray
import logging 
import torch

reduce_sequence = 0

def all_reduce_tensors(tensors):
  global reduce_sequence
  logging.info(f"received tensors {tensors}")
  for t in tensors:
    assert t.is_cuda
  reduce_sequence += 1 
  ray.get([all_reduce_impl(t, reduce_sequence) for t in tensors])


def all_reduce_impl(tensor, sequence):
  reducer_name = f"cli:{ray.get_runtime_context().get_node_id()}:{tensor.get_device()}" 
  logging.info(f"sending {tensor}, sequence: {sequence} to reducer: {reducer_name}")
  actor = ray.get_actor(reducer_name)
  return actor.allreduce.remote(tensor, sequence)
