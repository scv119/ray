# flake8: noqa

# fmt: off
# __tasks_start__

import ray
import time


# A regular Python function.
def normal_function():
    return 1


# By adding the `@ray.remote` decorator, a regular Python function
# becomes a Ray remote function.
@ray.remote
def my_function():
    return 1


# To invoke this remote function, use the `remote` method.
# This will immediately return an object ref (a future) and then create
# a task that will be executed on a worker process.
obj_ref = my_function.remote()

# The result can be retrieved with ``ray.get``.
assert ray.get(obj_ref) == 1


@ray.remote
def slow_function():
    time.sleep(10)
    return 1


# Ray tasks are executed in parallel.
# All computation is performed in the background, driven by Ray's internal event loop.
for _ in range(4):
    # This doesn't block.
    slow_function.remote()

# __tasks_end__
# fmt: on

# fmt: off
# __pass_by_ref_start__
@ray.remote
def function_with_an_argument(value):
    return value + 1


obj_ref1 = my_function.remote()
assert ray.get(obj_ref1) == 1

# You can pass an object ref as an argument to another Ray task.
obj_ref2 = function_with_an_argument.remote(obj_ref1)
assert ray.get(obj_ref2) == 2

# __pass_by_ref_end__
# fmt: on

# fmt: off
# __wait_start__

object_refs =  [slow_function.remote() for _ in range(2)]
# Return as soon as one of the tasks finished execution.
ready_refs, remaining_refs = ray.wait(object_refs, num_returns=1, timeout=None)

# __wait_end__
# fmt: on

# fmt: off
# __multiple_returns_start__

# By default, a Ray task only returns a single Object Ref.
@ray.remote
def return_single():
    return 0, 1, 2


object_ref = return_single.remote()
assert ray.get(object_ref) == (0, 1, 2)


# However, you can configure Ray tasks to return multiple Object Refs.
@ray.remote(num_returns=3)
def return_multiple():
    return 0, 1, 2


object_ref0, object_ref1, object_ref2 = return_multiple.remote()
assert ray.get(object_ref0) == 0
assert ray.get(object_ref1) == 1
assert ray.get(object_ref2) == 2

# __multiple_returns_end__
# fmt: on 


# fmt: off
# __generator_start__
@ray.remote(num_returns=3)
def return_multiple_as_generator():
    for i in range(3):
        yield i

# NOTE: Similar to normal functions, these objects will not be available
# until the full task is complete and all returns have been generated.
a, b, c = return_multiple_as_generator.remote()

# __generator_end__
# fmt: on

# fmt: off
# __cancel_start__
@ray.remote
def blocking_operation():
    time.sleep(10e6)

obj_ref = blocking_operation.remote()
ray.cancel(obj_ref)

try:
    ray.get(obj_ref)
except ray.exceptions.TaskCancelledError:
    print("Object reference was cancelled.")
# __cancel_end__
# fmt: on
