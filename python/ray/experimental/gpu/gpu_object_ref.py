class GpuObjectRef:
    def __init__(self, id, src_rank, shape, dtype):
        self.id = id
        self.src_rank = src_rank
        self.shape = shape
        self.dtype = dtype
