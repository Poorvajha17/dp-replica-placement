from models.data_grid import DataGridTree, ReplicaSolution
import random
from typing import Set

class ProportionalPlacement:
    """
    Proportional algorithm from reference [3] in the paper
    Used for comparison with our DP algorithm
    """
    
    def __init__(self, tree: DataGridTree):
        self.tree = tree
    
    def solve(self, k: int, max_workload: float = float('inf')) -> ReplicaSolution:
        """
        Simple proportional placement based on read requests
        This is a simplified version for comparison
        """
        # Start with root having a replica
        replica_set = {self.tree.root_id}
        
        # Calculate total read requests per subtree
        node_requests = {}
        for node_id in self.tree.nodes:
            node_requests[node_id] = self._compute_total_requests(node_id)
        
        # Sort nodes by total requests (descending)
        sorted_nodes = sorted(
            self.tree.nodes.keys(),
            key=lambda x: node_requests[x],
            reverse=True
        )
        
        # Add replicas to nodes with highest total requests
        for node_id in sorted_nodes:
            if len(replica_set) >= k:
                break
            if node_id not in replica_set:
                # Check if adding this replica violates workload constraints
                temp_set = replica_set | {node_id}
                if self._validate_workload(temp_set, max_workload):
                    replica_set.add(node_id)
        
        # Compute final cost and workloads
        total_cost = self.tree.compute_communication_cost(replica_set)
        node_workloads = {}
        for node_id in replica_set:
            node_workloads[node_id] = self.tree.compute_node_workload(node_id, replica_set)
        
        return ReplicaSolution(replica_set, total_cost, node_workloads)
    
    def _compute_total_requests(self, node_id: int) -> int:
        """Compute total read requests in subtree"""
        total = self.tree.nodes[node_id].read_requests
        for child_id in self.tree.get_children(node_id):
            total += self._compute_total_requests(child_id)
        return total
    
    def _validate_workload(self, replica_set: Set[int], max_workload: float) -> bool:
        """Check if workload constraints are satisfied"""
        if max_workload == float('inf'):
            return True
            
        for node_id in replica_set:
            workload = self.tree.compute_node_workload(node_id, replica_set)
            if workload > max_workload:
                return False
        return True