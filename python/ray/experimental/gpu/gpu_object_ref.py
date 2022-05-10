from typing import Tuple
import uuid
from cupy import cp


class GpuObjectRef:
    """Presents a reference to GPU buffer."""

    def __init__(self, id: uuid.UUID, src_rank: int, shape: Tuple, dtype: cp.dtype):
        self.id = id
        self.src_rank = src_rank
        self.shape = shape
        self.dtype = dtype
