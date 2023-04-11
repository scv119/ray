import logging
import socket
from collections import deque
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Any, Callable

import torch
from ray.experimental.parallel_ml.communicator.communicator import (
    FULLFILLED_FUTURE,
    Communicator,
)
from ray.experimental.parallel_ml.schedule import (
    Backward,
    Forward,
    Instruction,
    LoadBatch,
    Optimize,
    PrintOutput,
    ReceiveActivation,
    ReceiveGradient,
    Schedule,
    SendActivation,
    SendGradient,
)

logger = logging.getLogger(__name__)


@dataclass
class Config(object):
    """A config represents the configuration of a stage replica."""

    world_size: int
    rank: int
    input_tensor_shape: Any
    input_tensor_dtype: torch.Tensor.dtype
    device_name_builder: Callable[[], str]
    communicator_builder: Callable[[int, int, str], Communicator]
    model_builder: Callable[[], torch.nn.Module]
    data_loader_builder: Callable[[], torch.utils.data.DataLoader]
    optimizer_builder: Callable[[torch.nn.Module], torch.optim.Optimizer]


class ExecutionEngine:
    """A stage replica engine represents a physical replica in the pipeline stage.
    It follows a precomputed schedule to schedule until reconfigured.
    """

    def __init__(self, schedule: Schedule, config: Config, is_training: bool = False):
        self.is_training = is_training
        self.input_queue = deque()
        self.output_queue = deque()

        self.schedule = schedule
        self.execution_lock = Lock()
        self.stop = False
        self.config = config
        self.communicator_master_address = None

        # The following fields are only used for training
        # gradient if we are doing training.
        self.input_gradient = deque()
        self.input_gradient_tensor_shape = None
        self.input_gradient_tensor_dtype = None
        self.output_gradient = deque()
        self.forward_cache = {}
        self.forward_counter = 0
        self.backward_counter = 0

    def _initialize_config(self, config: Config):
        self.input_tensor_shape = config.input_tensor_shape
        self.input_tensor_dtype = config.input_tensor_dtype
        self.device = torch.device(config.device_name_builder())
        self.dist = config.communicator_builder(
            config.world_size, config.rank, self.communicator_master_address
        )
        self.model = config.model_builder().to(self.device)
        if not self.is_training:
            self.model.eval()
        self.data_loader = config.data_loader_builder()
        self.optimizer = config.optimizer_builder(self.model)

    def get_address(self):
        """Get the address of the engine."""
        return socket.gethostname()

    def start(self, master_address: str):
        """Start the engine execution"""
        self.communicator_master_address = master_address
        self._initialize_config(self.config)
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
        for instructions in self.schedule.steps():
            for instruction in instructions:
                self._execute_step(instruction)

    def _execute_step(self, instruction: Instruction):
        logger.info(f"Executing instruction {instruction}")
        if isinstance(instruction, SendActivation):
            self._send_activation(instruction)
        elif isinstance(instruction, ReceiveActivation):
            self._receive_activation(instruction)
        elif isinstance(instruction, Forward):
            self._forward(instruction)
        elif isinstance(instruction, PrintOutput):
            self._print_output(instruction)
        elif isinstance(instruction, LoadBatch):
            self._load_batch(instruction)
        elif isinstance(instruction, SendGradient):
            self._send_gradient(instruction)
        elif isinstance(instruction, ReceiveGradient):
            self._receive_gradient(instruction)
        elif isinstance(instruction, Optimize):
            self._optimize(instruction)
        elif isinstance(instruction, Backward):
            self._backward(instruction)

    def _send_activation(self, instruction: SendActivation):
        for _ in range(instruction.count):
            self.dist.send(
                self.output_queue.popleft(), instruction.dest_rank, async_op=True
            )
            # TODO: do we need to wait for the future to be completed?

    def _receive_activation(self, instruction: ReceiveActivation):
        for _ in range(instruction.count):
            tensor = torch.ones(()).new_empty(
                size=self.input_tensor_shape,
                dtype=self.input_tensor_dtype,
                device=self.device,
            )
            future = self.dist.recv(tensor, instruction.src_rank, async_op=True)
            self.input_queue.append((tensor, future))

    def _forward(self, instruction: Instruction):
        for _ in range(instruction.count):
            tensor, future = self.input_queue.popleft()
            future.wait()
            output = self.model.forward(tensor)

            if self.is_training:
                if self.forward_counter == 0:
                    self.received_gradient_tensor_shape = output.shape
                    self.received_gradient_tensor_dtype = output.dtype
                self.forward_cache[self.forward_counter] = (tensor, output)
                self.forward_counter += 1

            self.output_queue.append(output)

    def _load_batch(self, instruction: Instruction):
        for _ in range(instruction.count):
            tensor = torch.ones(()).new_empty(
                size=self.input_tensor_shape,
                dtype=self.input_tensor_dtype,
                device=self.device,
            )
            self.data_loader.next_batch(tensor)
            self.input_queue.append((tensor, FULLFILLED_FUTURE))

    def _print_output(self, instruction: Instruction):
        for _ in range(instruction.count):
            logger.info(self.output_queue.popleft())

    def _send_gradient(self, instruction: SendGradient):
        for _ in range(instruction.count):
            self.dist.send(
                self.output_gradient.popleft(), instruction.dest_rank, async_op=True
            )

    def _receive_gradient(self, instruction: ReceiveGradient):
        for _ in range(instruction.count):
            tensor = torch.ones(()).new_empty(
                size=self.received_gradient_tensor_shape,
                dtype=self.received_gradient_tensor_dtype,
                device=self.device,
            )
            future = self.dist.recv(tensor, instruction.src_rank, async_op=True)
            self.input_gradient.append((tensor, future))

    def _optimize(self, instruction: Optimize):
        # TODO: this probably needs to be changed
        self.optimizer.step()

    def _backward(self, instruction: Backward):
        for _ in range(instruction.count):
            tensor, future = self.input_gradient.popleft()
            future.wait()
            input, output = self.forward_cache.pop(self.backward_counter)
            torch.autograd.backward(tensors=output, grad_tensors=tensor)
            self.output_gradient.append(input.grad)
            # TODO: do we need to do something for optimize?
            self.backward_counter += 1
