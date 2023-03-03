# file name: setup.py
import os
import sys

import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["./src/naive_process_group.cc"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

cxx_flags = []
cxx_flags.append("-DUSE_C10D_NCCL")

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="ray_collectives",
        sources=sources,
        include_dirs=include_dirs,
        with_cuda=True,
        extra_compile_args={"cxx": cxx_flags, "nvcc": cxx_flags},
    )
else:
    module = cpp_extension.CppExtension(
        name="ray_collectives",
        sources=sources,
        include_dirs=include_dirs,
        with_cuda=True,
        extra_compile_args={"cxx": cxx_flags, "nvcc": cxx_flags},
    )

setup(
    name="ray_collectives",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
