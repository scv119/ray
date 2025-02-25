import copy
from typing import Any, Dict, List

from ray.autoscaler._private.util import hash_runtime_conf
from ray.core.generated.instance_manager_pb2 import Instance


class NodeProviderConfig(object):
    """
    NodeProviderConfig is the helper class to provide instance
    related configs.
    """

    def __init__(self, node_configs: Dict[str, Any]) -> None:
        self._sync_continuously = False
        self.update_configs(node_configs)

    def update_configs(self, node_configs: Dict[str, Any]) -> None:
        self._node_configs = node_configs
        self._calculate_hashes()
        self._sync_continuously = self._node_configs.get(
            "generate_file_mounts_contents_hash", True
        )

    def _calculate_hashes(self) -> None:
        self._runtime_hash, self._file_mounts_contents_hash = hash_runtime_conf(
            self._node_configs["file_mounts"],
            self._node_configs["cluster_synced_files"],
            [
                self._node_configs["worker_setup_commands"],
                self._node_configs["worker_start_ray_commands"],
            ],
            generate_file_mounts_contents_hash=self._node_configs.get(
                "generate_file_mounts_contents_hash", True
            ),
        )

    def get_node_config(self, instance_type_name: str) -> Dict[str, Any]:
        return copy.deepcopy(
            self._node_configs["available_node_types"][instance_type_name][
                "node_config"
            ]
        )

    def get_docker_config(self, instance_type_name: str) -> Dict[str, Any]:
        if "docker" not in self._node_configs:
            return {}
        docker_config = copy.deepcopy(self._node_configs.get("docker", {}))
        node_specific_docker_config = self._node_configs["available_node_types"][
            instance_type_name
        ].get("docker", {})
        docker_config.update(node_specific_docker_config)
        return docker_config

    def get_worker_start_ray_commands(self, instance: Instance) -> List[str]:
        if (
            instance.num_successful_updates > 0
            and not self._node_config_provider.restart_only
        ):
            return []
        return self._node_configs["worker_start_ray_commands"]

    def get_worker_setup_commands(self, instance: Instance) -> List[str]:
        if (
            instance.num_successful_updates > 0
            and self._node_config_provider.restart_only
        ):
            return []

        return self._node_configs["available_node_types"][instance.name][
            "worker_setup_commands"
        ]

    def get_node_type_specific_config(
        self, instance_type_name: str, config_name: str
    ) -> Any:
        config = self._node_config_provider.get_config(config_name)
        node_specific_config = self._node_configs["available_node_types"][
            instance_type_name
        ]
        if config_name in node_specific_config:
            config = node_specific_config[config_name]
        return config

    def get_config(self, config_name, default=None) -> Any:
        return self._node_configs.get(config_name, default)

    @property
    def restart_only(self) -> bool:
        return self._node_configs.get("restart_only", False)

    @property
    def no_restart(self) -> bool:
        return self._node_configs.get("no_restart", False)

    @property
    def runtime_hash(self) -> str:
        return self._runtime_hash

    @property
    def file_mounts_contents_hash(self) -> str:
        return self._file_mounts_contents_hash
