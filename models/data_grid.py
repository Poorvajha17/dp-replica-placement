from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class Node:
    id: int
    read_requests: int = 0

@dataclass
class ReplicaSolution:
    replica_nodes: Set[int]
    total_communication_cost: float
    node_workloads: Dict[int, float]
    is_feasible: bool = True

    def __str__(self):
        return (f"Replica nodes: {sorted(self.replica_nodes)}\n"
                f"Total cost: {self.total_communication_cost:.2f}\n"
                f"Max workload: {max(self.node_workloads.values()):.2f}")

class DataGridTree:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], float] = {}
        self.root_id: Optional[int] = None
        self._graph = nx.DiGraph()

    def add_node(self, node_id: int, read_requests: int = 0):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id, read_requests)
            self._graph.add_node(node_id, read_requests=read_requests)
        return self.nodes[node_id]

    def add_edge(self, parent_id: int, child_id: int, distance: float):
        self.add_node(parent_id)
        self.add_node(child_id)
        self.edges[(parent_id, child_id)] = distance
        self._graph.add_edge(parent_id, child_id, distance=distance)
        if self.root_id is None:
            self.root_id = parent_id
    
    def set_root(self, root_id: int):
        """Set the root node of the data grid tree"""
        if root_id not in self.nodes:
            raise ValueError(f"Root node {root_id} not found in tree")
        self.root_id = root_id

    def get_children(self, node_id: int) -> List[int]:
        return list(self._graph.successors(node_id))

    def get_parent(self, node_id: int) -> Optional[int]:
        preds = list(self._graph.predecessors(node_id))
        return preds[0] if preds else None

    def get_distance(self, parent_id: int, child_id: int) -> float:
        return self.edges.get((parent_id, child_id), 0.0)

    def compute_node_workload(self, node_id: int, replica_set: Set[int]) -> float:
        """Workload = node’s own requests + children’s if no replica"""
        node = self.nodes[node_id]
        children = self.get_children(node_id)
        if not children:
            return node.read_requests
        workload = node.read_requests
        for c in children:
            if c not in replica_set:
                workload += self.compute_node_workload(c, replica_set)
        return workload

    def compute_communication_cost(self, replica_set: Set[int]) -> float:
        """Total comm cost = Σ (read_requests × distance to nearest replica)"""
        total = 0.0
        for nid, node in self.nodes.items():
            replica = self._find_closest_replica(nid, replica_set)
            if replica is not None:
                total += node.read_requests * self._calculate_path_cost(nid, replica)
        return total

    def _find_closest_replica(self, node_id: int, replica_set: Set[int]) -> Optional[int]:
        current = node_id
        while current is not None:
            if current in replica_set:
                return current
            current = self.get_parent(current)
        return None

    def _calculate_path_cost(self, start_id: int, end_id: int) -> float:
        cost = 0.0
        current = start_id
        while current != end_id and current is not None:
            parent = self.get_parent(current)
            if parent is None:
                break
            cost += self.get_distance(parent, current)
            current = parent
        return cost

    def get_tree_height(self) -> int:
        def height(node_id):
            children = self.get_children(node_id)
            if not children:
                return 1
            return 1 + max(height(c) for c in children)
        return height(self.root_id) if self.root_id is not None else 0
