# file name: setup.py
import os
import sys

import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["./src/naive_process_group.cc"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="ray_collectives",
        sources=sources,
        include_dirs=include_dirs,
        with_cuda=True,
        extra_cflags=["-DUSE_C10D_NCCL"],
    )
else:
    module = cpp_extension.CppExtension(
        name="ray_collectives",
        sources=sources,
        include_dirs=include_dirs,
        with_cuda=True,
        extra_cflags=["-DUSE_C10D_NCCL"],
    )

setup(
    name="Ray-Collectives",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
