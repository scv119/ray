#!/usr/bin/env bash

# insall all unbounded dependencies in setup.py for ray core
for dependency in attrs jsonschema aiosignal frozenlist requests grpcio
do
    python -m pip install -U --pre $dependency
done