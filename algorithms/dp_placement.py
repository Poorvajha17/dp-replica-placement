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
    Implements the Dynamic Programming Based Replica Placement Algorithm
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

        if k < 1:
            raise ValueError("At least one replica required (root must have a replica)")

        self.dp_table.clear()

        tree_height = self.tree.get_tree_height()

        # Compute DP for entire tree
        root_solutions = self._compute_dp(self.tree.root_id, k, 0, tree_height)

        if not root_solutions:
            return ReplicaSolution(set(), float('inf'), {}, False)

        best_solution = min(root_solutions, key=lambda x: x.communication_cost)

        # Workload computation for replicas
        node_workloads = {}
        for node_id in best_solution.replica_set:
            node_workloads[node_id] = self.tree.compute_node_workload(
                node_id, best_solution.replica_set
            )

        # ✅ Ensure root has a replica
        if self.tree.root_id not in best_solution.replica_set:
            best_solution.replica_set.add(self.tree.root_id)

        return ReplicaSolution(
            replica_nodes=best_solution.replica_set,
            total_communication_cost=best_solution.communication_cost,
            node_workloads=node_workloads,
            is_feasible=True
        )

    # ----------------------------------------------------------------------
    def _compute_dp(self, node_id: int, k: int, l: int, max_l: int) -> List[DPSolution]:
        """
        Compute M(v, k, l) — Dynamic Programming recursive function.
        """
        key = (node_id, k, l)
        if key in self.dp_table:
            return self.dp_table[key]

        node = self.tree.nodes[node_id]
        children = self.tree.get_children(node_id)
        solutions = []

        # -----------------------------
        # Handle ROOT special case
        # -----------------------------
        is_root = (node_id == self.tree.root_id and l == 0)
        if is_root:
            # The root must have a replica, so we skip the “no-replica” branch
            l = 0  # Reset depth
        # -----------------------------

        # Case 1: Leaf node
        if not children:
            if k >= 1:  # Place replica at leaf
                workload = self.tree.compute_node_workload(node_id, {node_id})
                if workload <= self.max_workload:
                    solutions.append(DPSolution({node_id}, 0, workload))
            elif k == 0:  # No replica, must fetch from ancestor
                path_cost = self._calculate_path_cost_to_replica(node_id, l)
                cost = node.read_requests * path_cost
                solutions.append(DPSolution(set(), cost, node.read_requests))

        # Case 2: Internal node
        else:
            # --------------------------
            # Subcase 1: Replica at node
            # --------------------------
            if k >= 1:
                replica_workload = node.read_requests
                base_replicas = {node_id}

                # For each possible division of replicas between children
                if len(children) == 2:
                    left, right = children
                    for k_left in range(k):  # use k-1 left + right
                        k_right = k - 1 - k_left
                        if k_right < 0:
                            continue

                        left_solutions = self._compute_dp(left, k_left, 1, max_l)
                        right_solutions = self._compute_dp(right, k_right, 1, max_l)

                        for lsol in left_solutions:
                            for rsol in right_solutions:
                                new_replicas = base_replicas.union(lsol.replica_set).union(rsol.replica_set)
                                total_cost = lsol.communication_cost + rsol.communication_cost

                                workload = node.read_requests
                                if left not in lsol.replica_set:
                                    workload += lsol.workload
                                if right not in rsol.replica_set:
                                    workload += rsol.workload

                                if workload <= self.max_workload:
                                    solutions.append(DPSolution(new_replicas, total_cost, workload))

            # --------------------------
            # Subcase 2: No replica at node
            # --------------------------
            # Skip this if root, since root must have replica
            if not is_root:
                if len(children) == 2:
                    left, right = children
                    for k_left in range(k + 1):
                        k_right = k - k_left
                        left_solutions = self._compute_dp(left, k_left, l + 1, max_l)
                        right_solutions = self._compute_dp(right, k_right, l + 1, max_l)

                        for lsol in left_solutions:
                            for rsol in right_solutions:
                                new_replicas = lsol.replica_set.union(rsol.replica_set)
                                base_cost = lsol.communication_cost + rsol.communication_cost
                                path_cost = self._calculate_path_cost_to_replica(node_id, l)
                                total_cost = base_cost + node.read_requests * path_cost

                                workload = node.read_requests + lsol.workload + rsol.workload

                                if workload <= self.max_workload:
                                    solutions.append(DPSolution(new_replicas, total_cost, workload))

        filtered = self._filter_dominated(solutions)
        self.dp_table[key] = filtered
        return filtered

    # ----------------------------------------------------------------------
    def _calculate_path_cost_to_replica(self, node_id: int, l: int) -> float:
        """Calculate total communication cost to l-th ancestor"""
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
        """Remove dominated solutions (Pareto-efficient filtering)."""
        if not solutions:
            return []

        workload_map = {}
        for sol in solutions:
            wk = round(sol.workload, 2)
            if wk not in workload_map or sol.communication_cost < workload_map[wk].communication_cost:
                workload_map[wk] = sol

        return list(workload_map.values())
