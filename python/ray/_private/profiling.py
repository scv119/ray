import os
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Union

import ray


class _NullLogSpan:
    """A log span context manager that does nothing"""

    def __enter__(self):
        pass

    def __exit__(self, type, value, tb):
        pass


PROFILING_ENABLED = "RAY_PROFILING" in os.environ
NULL_LOG_SPAN = _NullLogSpan()

# Colors are specified at
# https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html.  # noqa: E501
_default_color_mapping = defaultdict(
    lambda: "generic_work",
    {
        "worker_idle": "cq_build_abandoned",
        "task": "rail_response",
        "task:deserialize_arguments": "rail_load",
        "task:execute": "rail_animation",
        "task:store_outputs": "rail_idle",
        "wait_for_function": "detailed_memory_dump",
        "ray.get": "good",
        "ray.put": "terrible",
        "ray.wait": "vsync_highlight_color",
        "submit_task": "background_memory_dump",
        "fetch_and_run_function": "detailed_memory_dump",
        "register_remote_function": "detailed_memory_dump",
    },
)


@dataclass(init=True)
class ChromeTracingCompleteEvent:
    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.lpfof2aylapb # noqa
    # The event categories. This is a comma separated list of categories
    # for the event. The categories can be used to hide events in
    # the Trace Viewer UI.
    cat: str
    # The string displayed on the event.
    name: str
    # The identifier for the group of rows that the event
    # appears in.
    pid: int
    # The identifier for the row that the event appears in.
    tid: int
    # The start time in microseconds.
    ts: int
    # The duration in microseconds.
    dur: int
    # This is the name of the color to display the box in.
    cname: str
    # The extra user-defined data.
    args: Dict[str, Union[str, int]]
    # The event type (X means the complete event).
    ph: str = "X"


@dataclass(init=True)
class ChromeTracingMetadataEvent:
    # https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#bookmark=id.iycbnb4z7i9g # noqa
    name: str
    # Metadata arguments. E.g., name: <metadata_name>
    args: Dict[str, str]
    # The process id of this event. In Ray, pid indicates the node.
    pid: int
    # The thread id of this event. In Ray, tid indicates each worker.
    tid: int = None
    # M means the metadata event.
    ph: str = "M"


def profile(event_type, extra_data=None):
    """Profile a span of time so that it appears in the timeline visualization.

    Note that this only works in the raylet code path.

    This function can be used as follows (both on the driver or within a task).

    .. code-block:: python
        import ray._private.profiling as profiling

        with profiling.profile("custom event", extra_data={'key': 'val'}):
            # Do some computation here.

    Optionally, a dictionary can be passed as the "extra_data" argument, and
    it can have keys "name" and "cname" if you want to override the default
    timeline display text and box color. Other values will appear at the bottom
    of the chrome tracing GUI when you click on the box corresponding to this
    profile span.

    Args:
        event_type: A string describing the type of the event.
        extra_data: This must be a dictionary mapping strings to strings. This
            data will be added to the json objects that are used to populate
            the timeline, so if you want to set a particular color, you can
            simply set the "cname" attribute to an appropriate color.
            Similarly, if you set the "name" attribute, then that will set the
            text displayed on the box in the timeline.

    Returns:
        An object that can profile a span of time via a "with" statement.
    """
    if not PROFILING_ENABLED:
        return NULL_LOG_SPAN
    worker = ray._private.worker.global_worker
    if worker.mode == ray._private.worker.LOCAL_MODE:
        return NULL_LOG_SPAN
    return worker.core_worker.profile_event(event_type.encode("ascii"), extra_data)


def chrome_tracing_dump(
    tasks: List[dict],
) -> str:
    """Generate a chrome/perfetto tracing dump using task events.

    Args:
        tasks: List of tasks generated by a state API list_tasks(detail=True).

    Returns:
        Json serialized dump to create a chrome/perfetto tracing.
    """
    # All events from given tasks.
    all_events = []

    # Chrome tracing doesn't have a concept of "node". Instead, we use
    # chrome tracing's pid == ray's node.
    # chrome tracing's tid == ray's process.
    # Note that pid or tid is usually integer, but ray's node/process has
    # ids in string.
    # Unfortunately, perfetto doesn't allow to have string as a value of pid/tid.
    # To workaround it, we use Metadata event from chrome tracing schema
    # (https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#heading=h.xqopa5m0e28f) # noqa
    # which allows pid/tid -> name mapping. In order to use this schema
    # we build node_ip/(node_ip, worker_id) -> arbitrary index mapping.

    # node ip address -> node idx.
    node_to_index = {}
    # Arbitrary index mapped to the ip address.
    node_idx = 0
    # (node index, worker id) -> worker idx
    worker_to_index = {}
    # Arbitrary index mapped to the (node index, worker id).
    worker_idx = 0

    for task in tasks:
        profiling_data = task.get("profiling_data", [])
        if profiling_data:
            node_ip_address = profiling_data["node_ip_address"]
            component_events = profiling_data["events"]
            component_type = profiling_data["component_type"]
            component_id = component_type + ":" + profiling_data["component_id"]

            if component_type not in ["worker", "driver"]:
                continue

            for event in component_events:
                extra_data = event["extra_data"]
                # Propagate extra data.
                extra_data["task_id"] = task["task_id"]
                extra_data["job_id"] = task["job_id"]
                extra_data["attempt_number"] = task["attempt_number"]
                extra_data["func_or_class_name"] = task["func_or_class_name"]
                extra_data["actor_id"] = task["actor_id"]
                event_name = event["event_name"]

                # build a id -> arbitrary index mapping
                if node_ip_address not in node_to_index:
                    node_to_index[node_ip_address] = node_idx
                    # Whenever new node ip is introduced, we increment the index.
                    node_idx += 1

                if (
                    node_to_index[node_ip_address],
                    component_id,
                ) not in worker_to_index:  # noqa
                    worker_to_index[
                        (node_to_index[node_ip_address], component_id)
                    ] = worker_idx  # noqa
                    worker_idx += 1

                # Modify the name with the additional user-defined extra data.
                cname = _default_color_mapping[event["event_name"]]
                name = event_name

                if "cname" in extra_data:
                    cname = _default_color_mapping[event["extra_data"]["cname"]]
                if "name" in extra_data:
                    name = extra_data["name"]

                new_event = ChromeTracingCompleteEvent(
                    cat=event_name,
                    name=name,
                    pid=node_to_index[node_ip_address],
                    tid=worker_to_index[(node_to_index[node_ip_address], component_id)],
                    ts=event["start_time"] * 1e3,
                    dur=(event["end_time"] * 1e3) - (event["start_time"] * 1e3),
                    cname=cname,
                    args=extra_data,
                )
                all_events.append(asdict(new_event))

    for node, i in node_to_index.items():
        all_events.append(
            asdict(
                ChromeTracingMetadataEvent(
                    name="process_name",
                    pid=i,
                    args={"name": f"Node {node}"},
                )
            )
        )

    for worker, i in worker_to_index.items():
        all_events.append(
            asdict(
                ChromeTracingMetadataEvent(
                    name="thread_name",
                    ph="M",
                    tid=i,
                    pid=worker[0],
                    args={"name": worker[1]},
                )
            )
        )

    # Handle task event disabled.
    return json.dumps(all_events)
