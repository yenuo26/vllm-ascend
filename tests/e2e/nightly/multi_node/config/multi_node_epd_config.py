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
                "d": []
            }

    def add_node_info(self, node_type: str, node_id: int, container_name="epd_vllm_ascend"):
        if node_type not in self.node_info:
            raise ValueError("node type can only be mooncake,proxy,e,pd,p,d")
        new_config = NodeConfig(node_id=node_id, container_name=container_name)
        self.node_info[node_type].append(new_config)
        print(f"add {node_type}: node_id={node_id}, container={container_name}")


    def get_all_info(self):
        return self.node_info


    def get_node_info(self, node_type: str, index: int = 0) -> Optional[NodeConfig]:
        if node_type in self.node_info and index < len(self.node_info[node_type]):
            return self.node_info[node_type][index]
        return None

