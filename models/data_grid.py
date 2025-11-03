from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class Node:
    id: int
    read_requests: int = 0
    workload: float = 0.0
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

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
        parent = self.add_node(parent_id)
        child = self.add_node(child_id)
        
        self.edges[(parent_id, child_id)] = distance
        self._graph.add_edge(parent_id, child_id, distance=distance)
        
        if self.root_id is None:
            self.root_id = parent_id
    
    def set_root(self, root_id: int):
        if root_id in self.nodes:
            self.root_id = root_id
    
    def get_children(self, node_id: int) -> List[int]:
        return list(self._graph.successors(node_id))
    
    def get_parent(self, node_id: int) -> Optional[int]:
        predecessors = list(self._graph.predecessors(node_id))
        return predecessors[0] if predecessors else None
    
    def get_distance(self, parent_id: int, child_id: int) -> float:
        return self.edges.get((parent_id, child_id), 0.0)
    
    def compute_node_workload(self, node_id: int, replica_set: Set[int]) -> float:
        """Compute workload for a node according to Eq.(1) in the paper"""
        node = self.nodes[node_id]
        children = self.get_children(node_id)
        
        if not children:  # Leaf node
            return node.read_requests
        
        workload = node.read_requests
        for child_id in children:
            if child_id not in replica_set:
                workload += self.compute_node_workload(child_id, replica_set)
        
        return workload
    
    def compute_communication_cost(self, replica_set: Set[int]) -> float:
        """Compute total communication cost according to Eq.(3) and Eq.(4)"""
        total_cost = 0.0
        
        for node_id, node in self.nodes.items():
            # Find closest replica
            closest_replica = self._find_closest_replica(node_id, replica_set)
            if closest_replica:
                path_cost = self._calculate_path_cost(node_id, closest_replica)
                total_cost += node.read_requests * path_cost
        
        return total_cost
    
    def _find_closest_replica(self, node_id: int, replica_set: Set[int]) -> Optional[int]:
        """Find the closest replica by traversing up the tree"""
        current = node_id
        while current is not None:
            if current in replica_set:
                return current
            current = self.get_parent(current)
        return None
    
    def _calculate_path_cost(self, start_id: int, end_id: int) -> float:
        """Calculate communication cost along path from start to end"""
        cost = 0.0
        current = start_id
        
        while current != end_id and current is not None:
            parent = self.get_parent(current)
            if parent is not None:
                cost += self.get_distance(parent, current)
                current = parent
            else:
                break
        
        return cost
    
    def validate_solution(self, replica_set: Set[int], max_workload: float) -> bool:
        """Validate if solution satisfies workload constraints"""
        for node_id in replica_set:
            workload = self.compute_node_workload(node_id, replica_set)
            if workload > max_workload:
                return False
        return True
    
    def get_tree_height(self) -> int:
        """Get the height of the tree"""
        if self.root_id is None:
            return 0
        
        def _get_height(node_id):
            children = self.get_children(node_id)
            if not children:
                return 1
            return 1 + max(_get_height(child) for child in children)
        
        return _get_height(self.root_id)