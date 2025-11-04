import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import random
from io import BytesIO
import json
import numpy as np

from models.data_grid import DataGridTree
from algorithms.dp_placement import DynamicProgrammingPlacer
from algorithms.proportional_placement import ProportionalPlacement
from examples.paper_example import create_paper_example_tree

class StreamlitVisualizer:
    def __init__(self):
        self.tree = None
        self.dp_solution = None
        self.prop_solution = None
        
    def run(self):
        st.set_page_config(
            page_title="Data Grid Replica Placement",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("📊 Data Grid Replica Placement Analysis")
        st.markdown("""
        Dynamic programming based replica placement algorithm for data grids.
        Based on the paper: *"A Dynamic Programming Based Replica Placement Algorithm in Data Grid"*
        """)
        
        # Initialize session state for tree persistence
        if 'custom_tree' not in st.session_state:
            st.session_state.custom_tree = None
        
        # Sidebar for inputs
        st.sidebar.header("Algorithm Parameters")
        
        # Tree selection
        tree_option = st.sidebar.selectbox(
            "Select Tree Structure",
            ["Paper Example", "Custom Tree"]
        )
        
        if tree_option == "Paper Example":
            self.tree = create_paper_example_tree()
            self.display_tree_summary()
            
        elif tree_option == "Custom Tree":
            self.tree = self.create_custom_tree_ui()
            if self.tree:
                self.display_tree_summary()
        
        if self.tree:
            # Algorithm parameters
            k = st.sidebar.slider("Number of Replicas (k)", 1, min(10, len(self.tree.nodes)), 3)
            max_workload = st.sidebar.slider("Maximum Workload (W)", 10, 200, 50)
            
            # Run algorithms
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Run DP Algorithm", type="primary"):
                    with st.spinner("Running Dynamic Programming Algorithm..."):
                        self.run_dp_algorithm(k, max_workload)
                    
            with col2:
                if st.button("Run Proportional Algorithm"):
                    with st.spinner("Running Proportional Algorithm..."):
                        self.run_proportional_algorithm(k, max_workload)
            
            # Display results
            if self.dp_solution or self.prop_solution:
                self.display_results()
                
            # Always show tree visualization
            self.visualize_tree()
            
            # Performance analysis section
            st.header("📈 Performance Analysis")
            self.run_performance_analysis()
    
    def display_tree_summary(self):
        """Display tree summary information"""
        if self.tree:
            tree_summary = self.tree.get_tree_summary()
            st.sidebar.success(f"**Tree Summary:**\n"
                            f"- Nodes: {tree_summary['total_nodes']}\n"
                            f"- Height: {tree_summary['tree_height']}\n"
                            f"- Total Requests: {tree_summary['total_read_requests']}\n"
                            f"- Binary: {tree_summary['is_binary']}")
    
    def create_custom_tree_ui(self):
        """UI for creating custom trees"""
        st.header("🔧 Custom Tree Builder")
        
        # Initialize session state for tree building
        if 'tree_nodes' not in st.session_state:
            st.session_state.tree_nodes = {}
        if 'tree_edges' not in st.session_state:
            st.session_state.tree_edges = []
        if 'root_node' not in st.session_state:
            st.session_state.root_node = None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Add Nodes")
            with st.form("add_node_form"):
                node_id = st.text_input("Node ID", value="A")
                read_requests = st.number_input("Read Requests", min_value=0, value=10)
                
                if st.form_submit_button("Add Node"):
                    if node_id and node_id.strip() and node_id not in st.session_state.tree_nodes:
                        st.session_state.tree_nodes[node_id] = read_requests
                        st.success(f"Added node {node_id} with {read_requests} requests")
                    else:
                        st.error(f"Node {node_id} already exists or invalid ID")
        
        with col2:
            st.subheader("Set Root Node")
            if st.session_state.tree_nodes:
                root_options = list(st.session_state.tree_nodes.keys())
                root_node = st.selectbox("Select Root Node", root_options)
                if st.button("Set as Root"):
                    st.session_state.root_node = root_node
                    st.success(f"Set {root_node} as root node")
            else:
                st.info("Add nodes first to set root")
        
        st.subheader("Add Edges")
        col3, col4 = st.columns(2)
        
        with col3:
            if st.session_state.tree_nodes:
                parent_node = st.selectbox("Parent Node", list(st.session_state.tree_nodes.keys()))
            else:
                parent_node = st.text_input("Parent Node", value="A")
        
        with col4:
            if st.session_state.tree_nodes:
                child_node = st.selectbox("Child Node", list(st.session_state.tree_nodes.keys()))
            else:
                child_node = st.text_input("Child Node", value="B")
        
        edge_distance = st.number_input("Edge Distance", min_value=0.0, value=1.0, step=0.1)
        
        col5, col6 = st.columns(2)
        with col5:
            if st.button("Add Edge"):
                if parent_node and child_node:
                    if parent_node == child_node:
                        st.error("Parent and child cannot be the same node")
                    elif any(p == parent_node and c == child_node for p, c, _ in st.session_state.tree_edges):
                        st.error("Edge already exists")
                    else:
                        st.session_state.tree_edges.append((parent_node, child_node, edge_distance))
                        st.success(f"Added edge {parent_node} -> {child_node} (distance: {edge_distance})")
        
        with col6:
            if st.button("Clear All"):
                st.session_state.tree_nodes = {}
                st.session_state.tree_edges = []
                st.session_state.root_node = None
                st.success("Cleared all nodes and edges")
        
        # Display current tree structure
        st.subheader("Current Tree Structure")
        if st.session_state.tree_nodes:
            st.write("**Nodes:**")
            nodes_df = pd.DataFrame([
                {"Node": node, "Read Requests": requests} 
                for node, requests in st.session_state.tree_nodes.items()
            ])
            st.dataframe(nodes_df, use_container_width=True)
            
            if st.session_state.tree_edges:
                st.write("**Edges:**")
                edges_df = pd.DataFrame([
                    {"Parent": parent, "Child": child, "Distance": dist} 
                    for parent, child, dist in st.session_state.tree_edges
                ])
                st.dataframe(edges_df, use_container_width=True)
            
            if st.session_state.root_node:
                st.write(f"**Root Node:** {st.session_state.root_node}")
        
        # Build and validate tree
        if st.session_state.tree_nodes and st.session_state.root_node:
            if st.button("Build and Validate Tree", type="primary"):
                tree = self.build_tree_from_session()
                if tree:
                    validation_result = self.validate_tree_structure(tree)
                    if validation_result["is_valid"]:
                        st.session_state.custom_tree = tree
                        st.success("✅ Tree built and validated successfully!")
                        return tree
                    else:
                        st.error(f"❌ Tree validation failed: {validation_result['message']}")
                        return None
        elif st.session_state.tree_nodes:
            st.warning("⚠️ Please set a root node before building the tree")
        
        # Return previously built tree if available
        return st.session_state.get('custom_tree', None)
    
    def build_tree_from_session(self):
        """Build DataGridTree from session state"""
        tree = DataGridTree()
        
        # Add all nodes
        for node_id, read_requests in st.session_state.tree_nodes.items():
            tree.add_node(node_id, read_requests)
        
        # Add all edges
        for parent, child, distance in st.session_state.tree_edges:
            tree.add_edge(parent, child, distance)
        
        # Set root
        if st.session_state.root_node:
            tree.set_root(st.session_state.root_node)
        
        return tree
    
    def validate_tree_structure(self, tree):
        """Validate that the graph is a proper tree"""
        # Check if it's a DAG and tree-structured
        G = tree._graph
        
        if not tree.root_id:
            return {"is_valid": False, "message": "No root node specified"}
        
        # Check for cycles
        try:
            if not nx.is_directed_acyclic_graph(G):
                return {"is_valid": False, "message": "Graph contains cycles"}
        except:
            return {"is_valid": False, "message": "Graph structure error"}
        
        # Check if it's a tree (exactly one path between root and any node)
        visited = set()
        stack = [tree.root_id]
        
        while stack:
            node = stack.pop()
            if node in visited:
                return {"is_valid": False, "message": f"Node {node} has multiple parents"}
            visited.add(node)
            stack.extend(tree.get_children(node))
        
        # Check if all nodes are reachable from root
        if len(visited) != len(tree.nodes):
            unreachable = set(tree.nodes.keys()) - visited
            return {"is_valid": False, "message": f"Unreachable nodes: {unreachable}"}
        
        return {"is_valid": True, "message": "Valid tree structure"}
    
    
    def run_dp_algorithm(self, k: int, max_workload: float):
        """Run DP algorithm and store results"""
        try:
            dp_placer = DynamicProgrammingPlacer(self.tree, max_workload)
            self.dp_solution = dp_placer.solve(k)
            st.success("✅ DP Algorithm completed successfully!")
        except Exception as e:
            st.error(f"❌ DP Algorithm failed: {str(e)}")
    
    def run_proportional_algorithm(self, k: int, max_workload: float):
        """Run proportional algorithm and store results"""
        try:
            prop_placer = ProportionalPlacement(self.tree)
            self.prop_solution = prop_placer.solve(k, max_workload)
            st.success("✅ Proportional Algorithm completed successfully!")
        except Exception as e:
            st.error(f"❌ Proportional Algorithm failed: {str(e)}")
    
    def display_results(self):
        """Display algorithm results in a comparative table"""
        st.header("📋 Algorithm Results")
        
        results_data = []
        
        if self.dp_solution and self.dp_solution.is_feasible:
            results_data.append({
                'Algorithm': 'Dynamic Programming',
                'Replica Nodes': str(sorted(self.dp_solution.replica_nodes)),
                'Total Cost': f"{self.dp_solution.total_communication_cost:.2f}",
                'Max Workload': f"{max(self.dp_solution.node_workloads.values()):.2f}",
                'Computation Time': f"{self.dp_solution.performance_stats.get('computation_time', 0):.4f}s",
                'DP Calls': self.dp_solution.performance_stats.get('dp_calls', 0),
                'Cache Hits': self.dp_solution.performance_stats.get('cache_hits', 0)
            })
        
        if self.prop_solution:
            results_data.append({
                'Algorithm': 'Proportional',
                'Replica Nodes': str(sorted(self.prop_solution.replica_nodes)),
                'Total Cost': f"{self.prop_solution.total_communication_cost:.2f}",
                'Max Workload': f"{max(self.prop_solution.node_workloads.values()):.2f}",
                'Computation Time': 'N/A',
                'DP Calls': 'N/A',
                'Cache Hits': 'N/A'
            })
        
        if results_data:
            df = pd.DataFrame(results_data)
            
            # Display metrics in columns
            if self.dp_solution and self.dp_solution.is_feasible and self.prop_solution:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    improvement = ((self.prop_solution.total_communication_cost - self.dp_solution.total_communication_cost) 
                                  / self.prop_solution.total_communication_cost * 100)
                    st.metric(
                        "DP Improvement", 
                        f"{improvement:.1f}%",
                        delta=f"Cost reduction"
                    )
                
                with col2:
                    st.metric(
                        "DP Total Cost", 
                        f"{self.dp_solution.total_communication_cost:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "DP Max Workload", 
                        f"{max(self.dp_solution.node_workloads.values()):.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Computation Time", 
                        f"{self.dp_solution.performance_stats.get('computation_time', 0):.4f}s"
                    )
            
            st.dataframe(df, use_container_width=True)
            
            # Cost comparison chart
            if len(results_data) > 1:
                fig = px.bar(
                    df, 
                    x='Algorithm', 
                    y='Total Cost',
                    title='Communication Cost Comparison',
                    color='Algorithm',
                    color_discrete_map={
                        'Dynamic Programming': '#1f77b4',
                        'Proportional': '#ff7f0e'
                    }
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="Algorithm",
                    yaxis_title="Total Communication Cost"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def visualize_tree(self):
        """Interactive tree visualization"""
        st.header("🌳 Tree Visualization")
        
        if self.tree is None:
            st.warning("No tree data available.")
            return
            
        # Create networkx graph for visualization
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_id, node in self.tree.nodes.items():
            G.add_node(node_id, read_requests=node.read_requests)
        
        # Add edges
        for (parent, child), distance in self.tree.edges.items():
            G.add_edge(parent, child, distance=distance)
        
        # Create Plotly figure with fixed layout properties
        pos = self._hierarchical_layout(G, self.tree.root_id)
        
        edge_x = []
        edge_y = []
        edge_labels = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_labels.append(f"d={self.tree.edges[edge]:.1f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        hover_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node label and hover info
            read_req = G.nodes[node]['read_requests']
            label = f"Node {node}"
            hover = f"Node {node}<br>Requests: {read_req}"
            
            # Color coding for replicas
            if (self.dp_solution and self.dp_solution.is_feasible and 
                node in self.dp_solution.replica_nodes):
                color = 'red'
                label += " 📦"
                hover += "<br>🔴 REPLICA"
                if node in self.dp_solution.node_workloads:
                    hover += f"<br>Workload: {self.dp_solution.node_workloads[node]:.1f}"
                size = 50
            else:
                color = 'lightblue'
                size = 40
                
            node_text.append(label)
            node_color.append(color)
            node_size.append(size)
            hover_text.append(hover)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='darkblue')
            ),
            textfont=dict(size=12, color='white'),
            hovertext=hover_text
        )
        
        # FIXED: Updated layout with correct properties
        layout = go.Layout(
            title=dict(
                text='Data Grid Tree Structure',
                x=0.5,
                xanchor='center'
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(
                text="🔴 Red nodes indicate replica placement",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=-0.1,
                xanchor='left', yanchor='top',
                font=dict(size=12, color='red')
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show workload distribution if solution exists
        if self.dp_solution and self.dp_solution.is_feasible:
            self.plot_workload_distribution()
    
    def plot_workload_distribution(self):
        """Plot workload distribution across replica nodes"""
        st.subheader("📊 Workload Distribution")
        
        if not self.dp_solution.replica_nodes:
            return
            
        nodes = sorted(self.dp_solution.replica_nodes)
        workloads = [self.dp_solution.node_workloads[node_id] for node_id in nodes]
        
        fig = px.bar(
            x=[f'Node {n}' for n in nodes],
            y=workloads,
            title='Workload Distribution Across Replica Nodes',
            labels={'x': 'Replica Nodes', 'y': 'Workload'},
            color=workloads,
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.update_traces(
            hovertemplate="<b>Node %{x}</b><br>Workload: %{y:.2f}<extra></extra>",
            text=workloads,
            texttemplate='%{text:.1f}',
            textposition='outside'
        )
        
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis_title="Workload",
            xaxis_title="Replica Nodes"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _hierarchical_layout(self, G, root):
        """Create hierarchical layout for tree visualization"""
        pos = {}
        levels = {}
        
        def _assign_levels(node, level=0):
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
            for child in G.successors(node):
                _assign_levels(child, level + 1)
        
        if root in G:
            _assign_levels(root)
        
        max_level = max(levels.keys()) if levels else 0
        for level, nodes in levels.items():
            y = 1.0 - (level / (max_level + 1)) if max_level > 0 else 0.5
            x_spacing = 1.0 / (len(nodes) + 1)
            for i, node in enumerate(nodes):
                x = (i + 1) * x_spacing
                pos[node] = (x, y)
        
        return pos
    
    def run_performance_analysis(self):
        """Interactive performance analysis"""
        st.subheader("🔍 Interactive Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Varying Replica Counts", "Varying Workload Constraints", "Comprehensive Algorithm Comparison"]
        )
        
        if analysis_type == "Varying Replica Counts":
            self.analyze_varying_replicas()
        elif analysis_type == "Varying Workload Constraints":
            self.analyze_varying_workload()
        else:
            self.analyze_comprehensive_comparison()
    
    def analyze_varying_replicas(self):
        """Analyze performance with varying replica counts"""
        st.write("Analyze how communication cost changes with different numbers of replicas:")
        
        max_k = st.slider("Maximum replicas to analyze", 2, min(15, len(self.tree.nodes)), 8)
        workload_constraint = st.slider("Workload constraint for analysis", 10, 200, 50)
        
        if st.button("Run Replica Analysis", type="secondary"):
            k_values = list(range(1, max_k + 1))
            dp_costs = []
            prop_costs = []
            dp_times = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, k in enumerate(k_values):
                status_text.text(f"Analyzing k={k}...")
                
                # DP Algorithm
                dp_placer = DynamicProgrammingPlacer(self.tree, workload_constraint)
                self.dp_solution = dp_placer.solve(k)
                if self.dp_solution.is_feasible:
                    dp_costs.append(self.dp_solution.total_communication_cost)
                    dp_times.append(self.dp_solution.performance_stats.get('computation_time', 0))
                else:
                    dp_costs.append(float('inf'))
                    dp_times.append(0)
                
                # Proportional Algorithm
                prop_placer = ProportionalPlacement(self.tree)
                self.prop_solution = prop_placer.solve(k, workload_constraint)
                prop_costs.append(self.prop_solution.total_communication_cost)
                
                progress_bar.progress((i + 1) / len(k_values))
            
            status_text.text("Analysis complete!")
            
            # Create comparison plot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Communication Cost vs Replicas', 'Computation Time vs Replicas'),
                horizontal_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=dp_costs, 
                    name='DP Algorithm', 
                    line=dict(color='blue', width=3),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=prop_costs, 
                    name='Proportional Algorithm', 
                    line=dict(color='red', width=3, dash='dash'),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=k_values, y=dp_times, 
                    name='DP Computation Time', 
                    line=dict(color='green', width=3),
                    mode='lines+markers'
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Number of Replicas (k)", row=1, col=1)
            fig.update_xaxes(title_text="Number of Replicas (k)", row=1, col=2)
            fig.update_yaxes(title_text="Total Communication Cost", row=1, col=1)
            fig.update_yaxes(title_text="Computation Time (seconds)", row=1, col=2)
            
            fig.update_layout(
                height=500,
                showlegend=True,
                title_text=f"Performance Analysis (Workload ≤ {workload_constraint})"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            analysis_data = {
                'Replicas': k_values,
                'DP Cost': dp_costs,
                'Proportional Cost': prop_costs,
                'DP Time (s)': dp_times
            }
            df_analysis = pd.DataFrame(analysis_data)
            st.dataframe(df_analysis, use_container_width=True)
    
    def analyze_varying_workload(self):
        """Analyze performance with varying workload constraints"""
        st.write("Analyze how workload constraints affect replica placement:")
        
        max_workload = st.slider("Maximum workload to analyze", 20, 200, 100)
        k_replicas = st.slider("Number of replicas", 1, min(10, len(self.tree.nodes)), 3)
        
        if st.button("Run Workload Analysis", type="secondary"):
            workload_values = list(range(20, max_workload + 1, 10))
            dp_costs = []
            prop_costs = []
            feasible_flags = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, workload in enumerate(workload_values):
                status_text.text(f"Analyzing workload={workload}...")
                
                # DP Algorithm
                dp_placer = DynamicProgrammingPlacer(self.tree, workload)
                self.dp_solution = dp_placer.solve(k_replicas)
                if self.dp_solution.is_feasible:
                    dp_costs.append(self.dp_solution.total_communication_cost)
                    feasible_flags.append(True)
                else:
                    dp_costs.append(float('inf'))
                    feasible_flags.append(False)
                
                # Proportional Algorithm
                prop_placer = ProportionalPlacement(self.tree)
                self.prop_solution = prop_placer.solve(k_replicas, workload)
                prop_costs.append(self.prop_solution.total_communication_cost)
                
                progress_bar.progress((i + 1) / len(workload_values))
            
            status_text.text("Analysis complete!")
            
            # Create comparison plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=workload_values, y=dp_costs,
                name='DP Algorithm',
                line=dict(color='blue', width=3),
                mode='lines+markers'
            ))
            
            fig.add_trace(go.Scatter(
                x=workload_values, y=prop_costs,
                name='Proportional Algorithm',
                line=dict(color='red', width=3, dash='dash'),
                mode='lines+markers'
            ))
            
            # Highlight feasible regions
            for i, feasible in enumerate(feasible_flags):
                if not feasible:
                    fig.add_vline(x=workload_values[i], line_dash="dot", 
                                line_color="red", opacity=0.5)
            
            fig.update_layout(
                title=f"Communication Cost vs Workload Constraint (k={k_replicas})",
                xaxis_title="Maximum Workload Constraint",
                yaxis_title="Total Communication Cost",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def analyze_comprehensive_comparison(self):
        """Comprehensive algorithm comparison"""
        st.header("🔬 Comprehensive Algorithm Comparison")
        
        st.markdown("""
        This analysis compares both algorithms across multiple scenarios to provide comprehensive performance insights.
        """)
        
        # Configuration for comprehensive analysis
        col1, col2 = st.columns(2)
        
        with col1:
            min_replicas = st.slider("Minimum Replicas", 1, 5, 1)
            max_replicas = st.slider("Maximum Replicas", 2, 10, 5)
        
        with col2:
            min_workload = st.slider("Minimum Workload", 10, 100, 20)
            max_workload = st.slider("Maximum Workload", 50, 200, 100)
            workload_step = st.slider("Workload Step", 5, 20, 10)
        
        if st.button("Run Comprehensive Comparison", type="primary"):
            with st.spinner("Running comprehensive analysis across multiple parameters..."):
                results = self.run_comprehensive_comparison_analysis(
                    min_replicas, max_replicas, min_workload, max_workload, workload_step
                )
                self.display_comprehensive_results(results)
    
    def run_comprehensive_comparison_analysis(self, min_replicas, max_replicas, min_workload, max_workload, workload_step):
        """Run comprehensive comparison analysis"""
        results = {
            'comparisons': []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        replica_values = list(range(min_replicas, max_replicas + 1))
        workload_values = list(range(min_workload, max_workload + 1, workload_step))
        
        total_iterations = len(replica_values) * len(workload_values)
        current_iteration = 0
        
        for k in replica_values:
            for workload in workload_values:
                current_iteration += 1
                status_text.text(f"Analyzing: Replicas {k}, Workload {workload}")
                
                # Run DP Algorithm
                try:
                    dp_placer = DynamicProgrammingPlacer(self.tree, workload)
                    dp_solution = dp_placer.solve(k)
                    dp_cost = dp_solution.total_communication_cost if dp_solution.is_feasible else float('inf')
                    dp_time = dp_solution.performance_stats.get('computation_time', 0) if dp_solution.is_feasible else 0
                    dp_feasible = dp_solution.is_feasible
                except Exception as e:
                    dp_cost = float('inf')
                    dp_time = 0
                    dp_feasible = False
                
                # Run Proportional Algorithm
                try:
                    prop_placer = ProportionalPlacement(self.tree)
                    prop_solution = prop_placer.solve(k, workload)
                    prop_cost = prop_solution.total_communication_cost
                    prop_feasible = True
                except Exception as e:
                    prop_cost = float('inf')
                    prop_feasible = False
                
                # Calculate improvement (only if both are feasible and have finite costs)
                if dp_feasible and prop_feasible and dp_cost < float('inf') and prop_cost < float('inf') and prop_cost > 0:
                    improvement = ((prop_cost - dp_cost) / prop_cost * 100)
                else:
                    improvement = 0
                
                results['comparisons'].append({
                    'replicas': k,
                    'workload': workload,
                    'dp_cost': dp_cost,
                    'prop_cost': prop_cost,
                    'dp_time': dp_time,
                    'improvement': improvement,
                    'dp_feasible': dp_feasible,
                    'prop_feasible': prop_feasible
                })
                
                progress_bar.progress(current_iteration / total_iterations)
        
        status_text.text("Comprehensive analysis complete!")
        return results

    def display_comprehensive_results(self, results):
        """Display comprehensive comparison results"""
        if not results or not results['comparisons']:
            st.error("No results to display. The analysis may have failed.")
            return
        
        df = pd.DataFrame(results['comparisons'])
        
        # Filter out invalid results for statistics
        valid_results = df[(df['dp_feasible']) & (df['prop_feasible']) & 
                          (df['dp_cost'] < float('inf')) & (df['prop_cost'] < float('inf'))]
        
        st.header("📊 Comprehensive Results Summary")
        
        if len(valid_results) == 0:
            st.error("No valid results found. The DP algorithm may not be finding feasible solutions.")
            st.info("Try increasing the workload constraint or using a different tree structure.")
            return
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_improvement = valid_results['improvement'].mean()
            st.metric("Average DP Improvement", f"{avg_improvement:.1f}%")
        
        with col2:
            success_rate = (len(valid_results) / len(df)) * 100
            st.metric("DP Success Rate", f"{success_rate:.1f}%")
        
        with col3:
            avg_dp_time = valid_results['dp_time'].mean()
            st.metric("Average DP Time", f"{avg_dp_time:.4f}s")
        
        with col4:
            best_improvement = valid_results['improvement'].max()
            st.metric("Best Improvement", f"{best_improvement:.1f}%")
        
        # Visualization 1: Improved Heatmap with Fallback Options
        st.subheader("📈 Improvement Analysis")
        
        # Try multiple visualization approaches
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Approach 1: Regular Heatmap (if we have enough data)
            if len(valid_results) >= 6:  # Need at least 6 data points for meaningful heatmap
                try:
                    # Create pivot table for heatmap
                    pivot_data = valid_results.pivot_table(
                        values='improvement', 
                        index='workload', 
                        columns='replicas',
                        aggfunc='mean'
                    ).fillna(0)  # Fill NaN with 0
                    
                    # Ensure we have a proper 2D matrix
                    if pivot_data.shape[0] > 1 and pivot_data.shape[1] > 1:
                        fig1 = px.imshow(
                            pivot_data,
                            title='DP Improvement Heatmap (%)',
                            labels=dict(x="Replicas", y="Workload", color="Improvement (%)"),
                            color_continuous_scale='RdYlGn',
                            aspect='auto',
                            text_auto=True  # Show values on heatmap
                        )
                        fig1.update_layout(
                            height=400,
                            xaxis_title="Number of Replicas",
                            yaxis_title="Workload Constraint"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.warning("Not enough variation in parameters for heatmap")
                        # Fallback to bar chart
                        self.create_improvement_barchart(valid_results)
                except Exception as e:
                    st.warning(f"Heatmap failed: {str(e)}")
                    self.create_improvement_barchart(valid_results)
            else:
                st.info("Not enough data points for heatmap")
                self.create_improvement_barchart(valid_results)
        
        with col_viz2:
            # Approach 2: 3D Surface Plot (alternative visualization)
            if len(valid_results) >= 8:
                try:
                    self.create_3d_surface_plot(valid_results)
                except:
                    # Fallback to scatter plot
                    self.create_improvement_scatter(valid_results)
            else:
                self.create_improvement_scatter(valid_results)
        
        # Visualization 2: Improvement Trends
        st.subheader("📊 Improvement Trends")
        
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            # Improvement by Replica Count
            improvement_by_replicas = valid_results.groupby('replicas')['improvement'].agg(['mean', 'std', 'count']).reset_index()
            if len(improvement_by_replicas) > 1:
                fig2 = px.line(
                    improvement_by_replicas,
                    x='replicas',
                    y='mean',
                    title='Average Improvement vs Replicas',
                    markers=True,
                    error_y='std' if 'std' in improvement_by_replicas.columns else None
                )
                fig2.update_layout(
                    xaxis_title="Number of Replicas",
                    yaxis_title="Average Improvement (%)",
                    height=350
                )
                # Add data point counts as annotations
                for i, row in improvement_by_replicas.iterrows():
                    fig2.add_annotation(
                        x=row['replicas'],
                        y=row['mean'],
                        text=f"n={row['count']}",
                        showarrow=False,
                        yshift=10
                    )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough replica variation for trend analysis")
        
        with col_trend2:
            # Improvement by Workload
            improvement_by_workload = valid_results.groupby('workload')['improvement'].agg(['mean', 'std', 'count']).reset_index()
            if len(improvement_by_workload) > 1:
                fig3 = px.line(
                    improvement_by_workload,
                    x='workload',
                    y='mean',
                    title='Average Improvement vs Workload',
                    markers=True,
                    error_y='std' if 'std' in improvement_by_workload.columns else None
                )
                fig3.update_layout(
                    xaxis_title="Workload Constraint",
                    yaxis_title="Average Improvement (%)",
                    height=350
                )
                # Add data point counts as annotations
                for i, row in improvement_by_workload.iterrows():
                    fig3.add_annotation(
                        x=row['workload'],
                        y=row['mean'],
                        text=f"n={row['count']}",
                        showarrow=False,
                        yshift=10
                    )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Not enough workload variation for trend analysis")
        
        # Visualization 3: Performance Matrix
        st.subheader("🎯 Performance Matrix")
        
        # Create a performance matrix showing success/failure patterns
        performance_matrix = df.pivot_table(
            values='dp_feasible', 
            index='workload', 
            columns='replicas',
            aggfunc=lambda x: '✅' if any(x) else '❌'
        ).fillna('❌')
        
        if not performance_matrix.empty:
            fig4 = go.Figure(data=go.Heatmap(
                z=[[1 if cell == '✅' else 0 for cell in row] for row in performance_matrix.values],
                x=performance_matrix.columns,
                y=performance_matrix.index,
                text=performance_matrix.values,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale=[[0, 'lightcoral'], [1, 'lightgreen']],
                showscale=False,
                hoverinfo='text'
            ))
            
            fig4.update_layout(
                title='Feasibility Matrix (✅ = DP Success, ❌ = DP Failure)',
                xaxis_title="Number of Replicas",
                yaxis_title="Workload Constraint",
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Visualization 4: Cost Comparison Scatter Plot
        st.subheader("💸 Cost Comparison")
        
        if len(valid_results) > 0:
            fig5 = go.Figure()
            
            fig5.add_trace(go.Scatter(
                x=valid_results['dp_cost'],
                y=valid_results['prop_cost'],
                mode='markers',
                name='DP vs Proportional Cost',
                marker=dict(
                    size=10,
                    color=valid_results['improvement'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Improvement %"),
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=[f"Replicas: {r}, Workload: {w}<br>DP: {dc:.1f}, Prop: {pc:.1f}<br>Improvement: {imp:.1f}%" 
                      for r, w, dc, pc, imp in zip(valid_results['replicas'], valid_results['workload'], 
                                                 valid_results['dp_cost'], valid_results['prop_cost'],
                                                 valid_results['improvement'])],
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Add diagonal line (y = x)
            max_cost = max(valid_results[['dp_cost', 'prop_cost']].max().max(), 1)
            fig5.add_trace(go.Scatter(
                x=[0, max_cost],
                y=[0, max_cost],
                mode='lines',
                name='Equal Cost',
                line=dict(dash='dash', color='gray', width=2)
            ))
            
            fig5.update_layout(
                title='DP Algorithm Cost vs Proportional Algorithm Cost',
                xaxis_title='DP Algorithm Cost',
                yaxis_title='Proportional Algorithm Cost',
                height=500,
                showlegend=True
            )
            
            # Add quadrant annotations
            fig5.add_annotation(x=max_cost*0.75, y=max_cost*0.25, 
                               text="DP Better", showarrow=False, font=dict(color="green", size=12))
            fig5.add_annotation(x=max_cost*0.25, y=max_cost*0.75, 
                               text="Proportional Better", showarrow=False, font=dict(color="red", size=12))
            
            st.plotly_chart(fig5, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Create a more informative display table
        display_df = df.copy()
        display_df['dp_status'] = display_df.apply(
            lambda row: f"{row['dp_cost']:.1f}" if row['dp_feasible'] and row['dp_cost'] < float('inf') else "FAILED", 
            axis=1
        )
        display_df['prop_status'] = display_df.apply(
            lambda row: f"{row['prop_cost']:.1f}" if row['prop_feasible'] and row['prop_cost'] < float('inf') else "FAILED", 
            axis=1
        )
        display_df['improvement_display'] = display_df.apply(
            lambda row: f"{row['improvement']:.1f}%" if row['dp_feasible'] and row['prop_feasible'] else "N/A", 
            axis=1
        )
        
        # Select only relevant columns for display
        display_columns = ['replicas', 'workload', 'dp_status', 'prop_status', 'improvement_display', 'dp_time']
        st.dataframe(display_df[display_columns], use_container_width=True)
        
        # Export results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv,
            file_name="comprehensive_comparison_results.csv",
            mime="text/csv"
        )

    def create_improvement_barchart(self, valid_results):
        """Create a bar chart when heatmap isn't possible"""
        if len(valid_results) == 0:
            return
        
        # Group by replicas and show average improvement
        if len(valid_results['replicas'].unique()) > 1:
            group_by = 'replicas'
            title = 'Average Improvement by Number of Replicas'
        else:
            group_by = 'workload'
            title = 'Average Improvement by Workload Constraint'
        
        avg_improvement = valid_results.groupby(group_by)['improvement'].mean().reset_index()
        
        fig = px.bar(
            avg_improvement,
            x=group_by,
            y='improvement',
            title=title,
            color='improvement',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            xaxis_title=group_by.capitalize(),
            yaxis_title="Average Improvement (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_improvement_scatter(self, valid_results):
        """Create a scatter plot showing improvement patterns"""
        if len(valid_results) == 0:
            return
        
        fig = px.scatter(
            valid_results,
            x='replicas',
            y='workload',
            size='improvement',
            color='improvement',
            title='Improvement Distribution',
            color_continuous_scale='RdYlGn',
            size_max=20,
            hover_data=['dp_cost', 'prop_cost']
        )
        fig.update_layout(
            xaxis_title="Number of Replicas",
            yaxis_title="Workload Constraint",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    def create_3d_surface_plot(self, valid_results):
        """Create a 3D surface plot for improvement visualization"""
        try:
            # Create pivot table for 3D surface
            pivot_improvement = valid_results.pivot_table(
                values='improvement', 
                index='workload', 
                columns='replicas',
                aggfunc='mean'
            ).fillna(0)
            
            if pivot_improvement.shape[0] > 1 and pivot_improvement.shape[1] > 1:
                fig = go.Figure(data=[go.Surface(
                    z=pivot_improvement.values,
                    x=pivot_improvement.columns,
                    y=pivot_improvement.index,
                    colorscale='RdYlGn'
                )])
                
                fig.update_layout(
                    title='3D Improvement Surface',
                    scene=dict(
                        xaxis_title='Replicas',
                        yaxis_title='Workload',
                        zaxis_title='Improvement (%)'
                    ),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"3D plot failed: {str(e)}")

def main():
    visualizer = StreamlitVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()