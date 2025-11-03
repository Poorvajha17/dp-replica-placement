import argparse
from models.data_grid import DataGridTree
from algorithms.dp_placement import DynamicProgrammingPlacer
from algorithms.proportional_placement import ProportionalPlacement
from utils.visualizer import TreeVisualizer
from examples.paper_example import create_paper_example_tree
import time

def main():
    parser = argparse.ArgumentParser(description='Replica Placement in Data Grid')
    parser.add_argument('--replicas', '-k', type=int, default=3, help='Number of replicas')
    parser.add_argument('--workload', '-w', type=float, default=50.0, help='Maximum workload')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize results')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare algorithms')
    parser.add_argument('--analysis', '-a', action='store_true', help='Run comprehensive analysis')
    
    args = parser.parse_args()
    
    # Create the example tree from the paper
    print("Creating data grid tree from paper example...")
    tree = create_paper_example_tree()
    
    print(f"Tree parameters:")
    print(f"- Total nodes: {len(tree.nodes)}")
    print(f"- Root node: {tree.root_id}")
    print(f"- Replicas to place: {args.replicas}")
    print(f"- Max workload: {args.workload}")
    print()
    
    # Run DP algorithm
    print("Running Dynamic Programming Algorithm...")
    dp_placer = DynamicProgrammingPlacer(tree, args.workload)
    
    start_time = time.time()
    dp_solution = dp_placer.solve(args.replicas)
    dp_time = time.time() - start_time
    
    print("DP Algorithm Results:")
    print(f"Solution found: {dp_solution.is_feasible}")
    if dp_solution.is_feasible:
        print(f"Replica nodes: {sorted(dp_solution.replica_nodes)}")
        print(f"Total communication cost: {dp_solution.total_communication_cost:.2f}")
        print(f"Maximum workload: {max(dp_solution.node_workloads.values()):.2f}")
        print(f"Computation time: {dp_time:.4f} seconds")
        
        # Show individual workloads
        print("\nWorkload distribution:")
        for node_id in sorted(dp_solution.replica_nodes):
            workload = dp_solution.node_workloads[node_id]
            print(f"  Node {node_id}: {workload:.2f}")
    print()
    
    # Compare with proportional algorithm
    if args.compare:
        print("Running Proportional Algorithm...")
        prop_placer = ProportionalPlacement(tree)
        
        start_time = time.time()
        prop_solution = prop_placer.solve(args.replicas, args.workload)
        prop_time = time.time() - start_time
        
        print("Proportional Algorithm Results:")
        print(f"Replica nodes: {sorted(prop_solution.replica_nodes)}")
        print(f"Total communication cost: {prop_solution.total_communication_cost:.2f}")
        print(f"Maximum workload: {max(prop_solution.node_workloads.values()):.2f}")
        print(f"Computation time: {prop_time:.4f} seconds")
        print()
        
        if dp_solution.is_feasible:
            improvement = ((prop_solution.total_communication_cost - dp_solution.total_communication_cost) 
                          / prop_solution.total_communication_cost * 100)
            print(f"DP algorithm improvement: {improvement:.2f}%")
            
            time_ratio = dp_time / prop_time if prop_time > 0 else float('inf')
            print(f"Time ratio (DP/Prop): {time_ratio:.2f}")
    
    # Run comprehensive analysis
    if args.analysis:
        print("\nRunning comprehensive analysis...")
        TreeVisualizer.plot_algorithm_comparison(tree, max_k=8, max_workload=args.workload)
    
    # Visualize results
    if args.visualize and dp_solution.is_feasible:
        print("Generating visualization...")
        TreeVisualizer.visualize_tree(
            tree, 
            dp_solution, 
            title=f"Optimal Replica Placement (k={args.replicas})"
        )
        
        # Also plot workload distribution
        TreeVisualizer.plot_workload_distribution(dp_solution)

if __name__ == "__main__":
    main()