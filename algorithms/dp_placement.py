from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from models.data_grid import DataGridTree, ReplicaSolution, Node
import math

@dataclass
class DPSolution:
    replica_set: Set[int]
    communication_cost: float
    workload: float
    
    def __lt__(self, other):
        return self.communication_cost < other.communication_cost

class DynamicProgrammingPlacer:
    """
    Implements the dynamic programming based replica placement algorithm
    from the paper "A Dynamic Programming Based Replica Placement Algorithm in Data Grid"
    """
    
    def __init__(self, tree: DataGridTree, max_workload: float):
        self.tree = tree
        self.max_workload = max_workload
        self.dp_table: Dict[Tuple[int, int, int], List[DPSolution]] = {}
        
    def solve(self, k: int) -> ReplicaSolution:
        """
        Solve the replica placement problem for k replicas
        """
        if self.tree.root_id is None:
            raise ValueError("Tree must have a root node")
        
        self.dp_table = {}
        
        # The root must contain a replica according to the paper
        if k < 1:
            raise ValueError("At least one replica required (root must have replica)")
        
        tree_height = self.tree.get_tree_height()
        
        # Compute solution for the entire tree
        root_solutions = self._compute_dp(
            self.tree.root_id, k, 0, tree_height
        )
        
        if not root_solutions:
            return ReplicaSolution(set(), float('inf'), {}, False)
        
        # Find best solution
        best_solution = min(root_solutions, key=lambda x: x.communication_cost)
        
        # Compute workloads for all replica nodes
        node_workloads = {}
        for node_id in best_solution.replica_set:
            node_workloads[node_id] = self.tree.compute_node_workload(
                node_id, best_solution.replica_set
            )
        
        return ReplicaSolution(
            replica_nodes=best_solution.replica_set,
            total_communication_cost=best_solution.communication_cost,
            node_workloads=node_workloads
        )
    
    def _compute_dp(self, node_id: int, k: int, l: int, max_l: int) -> List[DPSolution]:
        """
        Compute M(v, k, l) - dynamic programming recursive function
        """
        key = (node_id, k, l)
        
        if key in self.dp_table:
            return self.dp_table[key]
        
        node = self.tree.nodes[node_id]
        children = self.tree.get_children(node_id)
        solutions = []
        
        # Case 1: Leaf node
        if not children:
            if k == 1:  # Place replica at leaf
                workload = self.tree.compute_node_workload(node_id, {node_id})
                if workload <= self.max_workload:
                    solutions.append(DPSolution({node_id}, 0, workload))
            elif k == 0:  # No replica
                path_cost = self._calculate_path_cost_to_replica(node_id, l)
                cost = node.read_requests * path_cost
                solutions.append(DPSolution(set(), cost, node.read_requests))
        
        # Case 2: Internal node
        else:
            # Subcase 2.1: Place replica at current node
            if k >= 1:
                # For binary trees (simplified)
                if len(children) == 2:
                    left_child, right_child = children
                    for k_left in range(k):
                        k_right = k - 1 - k_left
                        if k_right < 0:
                            continue
                            
                        left_sols = self._compute_dp(left_child, k_left, 1, max_l)
                        right_sols = self._compute_dp(right_child, k_right, 1, max_l)
                        
                        for left_sol in left_sols:
                            for right_sol in right_sols:
                                new_set = {node_id}.union(left_sol.replica_set).union(right_sol.replica_set)
                                total_cost = left_sol.communication_cost + right_sol.communication_cost
                                
                                # Compute workload for current node
                                workload = node.read_requests
                                if left_child not in left_sol.replica_set:
                                    workload += left_sol.workload
                                if right_child not in right_sol.replica_set:
                                    workload += right_sol.workload
                                
                                if workload <= self.max_workload:
                                    solutions.append(DPSolution(new_set, total_cost, workload))
            
            # Subcase 2.2: No replica at current node
            for k_left in range(k + 1):
                k_right = k - k_left
                if k_right < 0:
                    continue
                
                if len(children) == 2:
                    left_child, right_child = children
                    left_sols = self._compute_dp(left_child, k_left, l + 1, max_l)
                    right_sols = self._compute_dp(right_child, k_right, l + 1, max_l)
                    
                    for left_sol in left_sols:
                        for right_sol in right_sols:
                            new_set = left_sol.replica_set.union(right_sol.replica_set)
                            base_cost = left_sol.communication_cost + right_sol.communication_cost
                            path_cost = self._calculate_path_cost_to_replica(node_id, l)
                            total_cost = base_cost + node.read_requests * path_cost
                            
                            workload = node.read_requests + left_sol.workload + right_sol.workload
                            
                            if workload <= self.max_workload:
                                solutions.append(DPSolution(new_set, total_cost, workload))
        
        # Filter dominated solutions
        filtered_solutions = self._filter_dominated(solutions)
        self.dp_table[key] = filtered_solutions
        return filtered_solutions
    
    def _calculate_path_cost_to_replica(self, node_id: int, l: int) -> float:
        """Calculate distance to l-th ancestor"""
        cost = 0.0
        current = node_id
        for _ in range(l):
            parent = self.tree.get_parent(current)
            if parent is None:
                break
            cost += self.tree.get_distance(parent, current)
            current = parent
        return cost
    
    def _filter_dominated(self, solutions: List[DPSolution]) -> List[DPSolution]:
        """Remove dominated solutions"""
        if not solutions:
            return []
        
        # Group by workload and keep minimum cost for each workload
        workload_map = {}
        for sol in solutions:
            workload_key = round(sol.workload, 2)
            if workload_key not in workload_map or sol.communication_cost < workload_map[workload_key].communication_cost:
                workload_map[workload_key] = sol
        
        return list(workload_map.values())