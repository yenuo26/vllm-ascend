from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class NodeConfig:
    node_id: int
    container_name: str

@dataclass
class ClusterManager:
    node_info: Dict[str, List[NodeConfig]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_info:
            self.node_info = {
                "e": [],
                "pd": [],
                "p": [],
                "d": [],
                "ds": []
            }

    def add_node_info(self, node_type: str, node_id: int, container_name="epd_vllm_ascend"):
        if node_type not in self.node_info:
            raise ValueError("node type can only be e,pd,p,d")
        new_config = NodeConfig(node_id=node_id, container_name=container_name)
        self.node_info[node_type].append(new_config)
        print(f"add {node_type}: node_id={node_id}, container={container_name}")


    def get_all_info(self):
        return self.node_info


    def get_node_info(self, node_type: str, index: int = 0) -> Optional[NodeConfig]:
        if node_type in self.node_info and index < len(self.node_info[node_type]):
            return self.node_info[node_type][index]
        return None



@dataclass
class NodeEnv:
    env_key: str
    env_value: str

@dataclass
class EnvManager:
    env_info: Dict[str, List[NodeEnv]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.env_info:
            self.env_info = {
                "e": [],
                "pd": [],
                "p": [],
                "d": [],
                "proxy": [],
                "ds": [],
                "common": []
            }

    def add_env(self, node_type: str, env_key: str, env_value: str, env_dict=None):
        if node_type not in self.env_info:
            raise ValueError("node type can only be e,pd,p,d,proxy,ds,common")
        if env_dict is None:
            env_list = list()
            if not isinstance(env_dict,list):
                env_list.append(env_dict)
            for envs in env_list:
                for key, value in envs.items():
                    new_env = NodeEnv(env_key=key, env_value=value)
                    self.env_info[node_type].append(new_env)
        else:
            new_env = NodeEnv(env_key=env_key, env_value=env_value)
            self.env_info[node_type].append(new_env)



    def get_all_env(self):
        return self.env_info


    def get_node_env(self, node_type: str, index: int = None):
        if node_type in self.env_info and index is None:
            return self.env_info[node_type]
        elif node_type in self.env_info and index < len(self.env_info[node_type]):
            return self.env_info[node_type][index]
        return None
