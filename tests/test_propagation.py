import unittest

import igraph as ig
import matplotlib.pyplot as plt

from figures.propagation import PropagationPlotFactory
from tests.conftest import populate_test_database


class PropagationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        populate_test_database('test_dataset')

    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database('test_dataset')

    def setUp(self):
        self.plot_factory = PropagationPlotFactory(available_datasets=['test_dataset'])

    def test_propagation_tree_original(self):
        graph = self.plot_factory.get_propagation_tree('test_dataset', '1112726971497230337')
        # Display the igraph graph with matplotlib
        layout = graph.layout(self.plot_factory.layout)
        fig, ax = plt.subplots()
        ig.plot(graph, layout=layout, target=ax)
        plt.show()




if __name__ == '__main__':
    unittest.main()
