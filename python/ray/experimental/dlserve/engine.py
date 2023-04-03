from abc import ABC
from collections import deque
from threading import Lock, Thread
from typing import Any, Callable

import torch
from ray.experimental.dlserve.communicator import TorchBasedCommunicator


class Instruction(ABC):
    """An instruction represents a single step in the execution schedule."""

    pass


class Send(Instruction):
    """Send data to dest rank."""

    def __init__(self, dest_rank: int, count: int = 1):
        self.dest_rank = dest_rank
        self.count = count


class Receive(Instruction):
    """Receive data from dest rank."""

    def __init__(self, src_rank: int, count: int = 1):
        self.src_rank = src_rank
        self.count = count


class Forward(Instruction):
    """Apply forward computation against the model."""

    def __init__(self, count: int = 1):
        self.count = count


class Schedule(ABC):
    """A schedule represents the execution schedule of a stage replica."""

    def steps(self):
        """Yield a list of :class:`Instructions` for each step in the schedule."""
        pass


class Config:
    def __init__(
        self,
        world_size: int,
        rank: int,
        input_tensor_shape: Any,
        input_tensor_dtype: torch.type,
        device_name: str,
        model_builder: Callable[[], torch.nn.Module],
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.input_tensor_shape = input_tensor_shape
        self.input_tensor_dtype = input_tensor_dtype
        self.device_name = device_name
        self.model_builder = model_builder


class ExecutionEngine:
    """A stage replica engine represents a physical replica in the pipeline stage."""

    def __init__(self, schedule: Schedule, config: Config):
        self.input_queue = deque()
        self.output_queue = deque()
        self.schedule = schedule
        self.execution_lock = Lock()
        self.stop = False
        self.config = config
        self._initialize_config(config)

    def _initialize_config(self, config: Config):
        self.input_tensor_shape = config.input_tensor_shape
        self.input_tensor_dtype = config.input_tensor_dtype
        self.cuda = torch.device(config.device_name)
        self.communicator = TorchBasedCommunicator(config.world_size, config.rank)
        self.model = config.model_builder().to(self.cuda)

    def start(self):
        """Start the engine execution"""
        self.thread = Thread(target=self._execute)
        self.thread.start()

    def stop(self):
        """Stop the engine if it's running."""
        with self.execution_lock:
            self.stop = True
        self.thread.join()

    def check_state(self):
        """Check the state of the engine."""
        pass

    def reconfigure(self, schedule: Schedule, config: Config):
        """Reconfgure the engine with a new schedule."""
        pass

    def _execute(self):
        with self.execution_lock:
            if self.stop:
                return
        for instruction in self.schedule.steps():
            self._execute_instruction(instruction)

    def _execute_step(self, instruction: Instruction):
        if isinstance(instruction, Send):
            for _ in range(instruction.count):
                self.communicator.send(
                    self.output_queue.popleft(), instruction.dest_rank, async_op=True
                )
                # TODO: do we need to wait for the future to be completed?
        elif isinstance(instruction, Receive):
            for _ in range(instruction.count):
                tensor = torch.new_empty(
                    size=self.input_tensor_shape,
                    dtype=self.input_tensor_dtype,
                    device=self.cuda,
                )
                future = self.communicator.recv(
                    tensor, instruction.src_rank, async_op=True
                )
                self.input_queue.append((tensor, future))
        if isinstance(instruction, Forward):
            for _ in range(instruction.count):
                tensor, future = self.input_queue.popleft()
                future.wait()
                self.output_queue.append(self.model.forward(tensor))
