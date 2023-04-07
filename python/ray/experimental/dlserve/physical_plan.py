from abc import ABC
from ray.dag import DAGNode
from ray.util.placement_group import PlacementGroup

class PhysicalPlan:
    def __init__(self) -> None:
        self.num_stages = []
        self.replica_schedules = []
        self.replica_configs = []

    def deploy(self):
        pass


class PhysicalPlanner(ABC):
    """The physical planner takes a logical plan (DAG) and number of
    availalble resources in the cluster, and generates a physical plan:
    i.e. number of replicas of each stage, and execution schedule of each replica."""

    def plan(self, logical_plan: DAGNode, pg: PlacementGroup) -> PhysicalPlan:
        pass


class SimplePhysicalPlanner(PhysicalPlanner):
    def plan(self, logical_plan: DAGNode, pg: PlacementGroup) -> PhysicalPlan:
        pass