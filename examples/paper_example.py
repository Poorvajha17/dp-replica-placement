from models.data_grid import DataGridTree

def create_paper_example_tree() -> DataGridTree:
    """Create the EXACT tree from Figure 4 in the paper with exact values"""
    tree = DataGridTree()
    
    # Build the tree structure from the paper with EXACT values
    # Level 0
    tree.add_node(0, read_requests=10)  # Root (A)
    
    # Level 1
    tree.add_edge(0, 1, 3)  # A->B, distance=3
    tree.add_edge(0, 2, 2)  # A->C, distance=2
    
    # Level 2  
    tree.add_edge(1, 3, 4)  # B->D, distance=4
    tree.add_edge(1, 4, 1)  # B->E, distance=1
    tree.add_edge(2, 5, 2)  # C->F, distance=2  
    tree.add_edge(2, 6, 3)  # C->G, distance=3
    
    # Level 3
    tree.add_edge(3, 7, 2)  # D->H, distance=2
    tree.add_edge(3, 8, 3)  # D->I, distance=3
    
    # Set EXACT read requests from paper
    tree.nodes[0].read_requests = 10  # A
    tree.nodes[1].read_requests = 8   # B
    tree.nodes[2].read_requests = 12  # C  
    tree.nodes[3].read_requests = 15  # D
    tree.nodes[4].read_requests = 8   # E
    tree.nodes[5].read_requests = 7   # F
    tree.nodes[6].read_requests = 9   # G
    tree.nodes[7].read_requests = 6   # H
    tree.nodes[8].read_requests = 5   # I
    
    tree.set_root(0)
    return tree