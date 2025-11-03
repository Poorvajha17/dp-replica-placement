import unittest
from models.data_grid import DataGridTree
from algorithms.dp_placement import DynamicProgrammingPlacer
from algorithms.proportional_placement import ProportionalPlacement

class TestReplicaPlacement(unittest.TestCase):
    
    def setUp(self):
        self.tree = DataGridTree()
        # Create a simple test tree
        self.tree.add_node(0, 5)
        self.tree.add_edge(0, 1, 2)
        self.tree.add_edge(0, 2, 3)
        self.tree.add_edge(1, 3, 1)
        self.tree.add_edge(1, 4, 2)
        self.tree.set_root(0)
    
    def test_dp_algorithm(self):
        placer = DynamicProgrammingPlacer(self.tree, 50)
        solution = placer.solve(2)
        self.assertTrue(solution.is_feasible)
        self.assertEqual(len(solution.replica_nodes), 2)
    
    def test_proportional_algorithm(self):
        placer = ProportionalPlacement(self.tree)
        solution = placer.solve(2, 50)
        self.assertEqual(len(solution.replica_nodes), 2)

if __name__ == '__main__':
    unittest.main()