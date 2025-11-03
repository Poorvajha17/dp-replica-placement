from algorithms.dp_placement import DynamicProgrammingPlacer
from algorithms.proportional_placement import ProportionalPlacement
import matplotlib.pyplot as plt
import networkx as nx
from models.data_grid import DataGridTree, ReplicaSolution
import math

class TreeVisualizer:
    """Visualize the tree and replica placement solutions without pygraphviz"""
    
    @staticmethod
    def visualize_tree(tree: DataGridTree, solution: ReplicaSolution = None, title: str = "Data Grid Tree"):
        plt.figure(figsize=(12, 8))
        
        # Use hierarchical layout instead of graphviz
        pos = TreeVisualizer._hierarchical_layout(tree)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        for node_id in tree.nodes:
            if solution and node_id in solution.replica_nodes:
                node_colors.append('red')  # Replica nodes in red
                node_sizes.append(1000)
            else:
                node_colors.append('lightblue')
                node_sizes.append(800)
        
        nx.draw_networkx_nodes(tree._graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, edgecolors='black')
        
        # Draw edges
        nx.draw_networkx_edges(tree._graph, pos, alpha=0.6, arrows=True, 
                              edge_color='gray', width=2)
        
        # Labels with read requests
        labels = {}
        for node_id, node in tree.nodes.items():
            labels[node_id] = f"{node_id}\nreq:{node.read_requests}"
        
        nx.draw_networkx_labels(tree._graph, pos, labels, font_size=8, font_weight='bold')
        
        # Edge labels with distances
        edge_labels = {(u, v): f"d:{d}" for (u, v), d in tree.edges.items()}
        nx.draw_networkx_edge_labels(tree._graph, pos, edge_labels, font_size=7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        if solution:
            plt.suptitle(
                f"Replicas: {sorted(solution.replica_nodes)} | "
                f"Cost: {solution.total_communication_cost:.2f} | "
                f"Max Workload: {max(solution.node_workloads.values()):.2f}",
                fontsize=12
            )
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _hierarchical_layout(tree: DataGridTree):
        """Create a hierarchical layout manually"""
        pos = {}
        levels = TreeVisualizer._assign_levels(tree)
        
        max_nodes_in_level = max(len(nodes) for nodes in levels.values())
        level_height = 1.0 / (len(levels) + 1)
        
        for level, nodes in levels.items():
            y = 1.0 - (level * level_height)
            x_spacing = 1.0 / (len(nodes) + 1)
            for i, node_id in enumerate(nodes):
                x = (i + 1) * x_spacing
                pos[node_id] = (x, y)
        
        return pos
    
    @staticmethod
    def _assign_levels(tree: DataGridTree):
        """Assign nodes to levels based on depth from root"""
        levels = {}
        
        def _assign_node_level(node_id, level):
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)
            
            for child_id in tree.get_children(node_id):
                _assign_node_level(child_id, level + 1)
        
        if tree.root_id is not None:
            _assign_node_level(tree.root_id, 0)
        
        return levels
    
    @staticmethod
    def plot_cost_comparison(costs_dp: list, costs_prop: list, k_values: list):
        """Plot comparison of communication costs"""
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, costs_dp, 'b-o', label='DP Algorithm', linewidth=2, markersize=8)
        plt.plot(k_values, costs_prop, 'r--s', label='Proportional Algorithm', linewidth=2, markersize=8)
        plt.xlabel('Number of Replicas (k)', fontsize=12)
        plt.ylabel('Total Communication Cost', fontsize=12)
        plt.title('Communication Cost Comparison: DP vs Proportional Algorithm', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_workload_distribution(solution: ReplicaSolution):
        """Plot workload distribution across replica nodes"""
        if not solution.replica_nodes:
            return
            
        plt.figure(figsize=(10, 6))
        nodes = sorted(solution.replica_nodes)
        workloads = [solution.node_workloads[node_id] for node_id in nodes]
        
        bars = plt.bar(range(len(nodes)), workloads, color='lightcoral', alpha=0.7)
        plt.xlabel('Replica Nodes', fontsize=12)
        plt.ylabel('Workload', fontsize=12)
        plt.title('Workload Distribution Across Replica Nodes', fontsize=14, fontweight='bold')
        plt.xticks(range(len(nodes)), [f'Node {n}' for n in nodes])
        
        # Add value labels on bars
        for bar, workload in zip(bars, workloads):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{workload:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_algorithm_comparison(tree: DataGridTree, max_k: int = 10, max_workload: float = 50):
        """Comprehensive comparison of both algorithms"""
        k_values = list(range(1, max_k + 1))
        dp_costs = []
        prop_costs = []
        dp_times = []
        prop_times = []
        
        import time
        
        print("Running comprehensive algorithm comparison...")
        for k in k_values:
            # DP Algorithm
            dp_placer = DynamicProgrammingPlacer(tree, max_workload)
            start_time = time.time()
            dp_solution = dp_placer.solve(k)
            dp_time = time.time() - start_time
            
            # Proportional Algorithm  
            prop_placer = ProportionalPlacement(tree)
            start_time = time.time()
            prop_solution = prop_placer.solve(k, max_workload)
            prop_time = time.time() - start_time
            
            if dp_solution.is_feasible:
                dp_costs.append(dp_solution.total_communication_cost)
            else:
                dp_costs.append(float('inf'))
                
            prop_costs.append(prop_solution.total_communication_cost)
            dp_times.append(dp_time)
            prop_times.append(prop_time)
            
            print(f"k={k}: DP Cost={dp_costs[-1]:.2f}, Prop Cost={prop_costs[-1]:.2f}")
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cost comparison
        ax1.plot(k_values, dp_costs, 'b-o', label='DP Algorithm', linewidth=2, markersize=6)
        ax1.plot(k_values, prop_costs, 'r--s', label='Proportional Algorithm', linewidth=2, markersize=6)
        ax1.set_xlabel('Number of Replicas (k)')
        ax1.set_ylabel('Total Communication Cost')
        ax1.set_title('Communication Cost Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time comparison
        ax2.plot(k_values, dp_times, 'g-^', label='DP Algorithm Time', linewidth=2, markersize=6)
        ax2.plot(k_values, prop_times, 'm--d', label='Proportional Algorithm Time', linewidth=2, markersize=6)
        ax2.set_xlabel('Number of Replicas (k)')
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('Computation Time Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return k_values, dp_costs, prop_costs, dp_times, prop_times