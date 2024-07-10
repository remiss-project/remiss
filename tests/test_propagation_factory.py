import random
import unittest

import igraph as ig
import pandas as pd
from pymongo import MongoClient

from figures.propagation import PropagationPlotFactory


class PropagationFactoryTestCase(unittest.TestCase):
    def setUp(self):
        self.propagation_factory = PropagationPlotFactory()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.test_user_id = '999321854'
        self.test_tweet_id = '1167078759280889856'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def _test_original_plot(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        # layout = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7), (8, 8, 8), (9, 9, 9)]
        layout = graph.layout('kk', dim=3)
        N = len(layout)
        Edges = graph.get_edgelist()
        Xn = [layout[k][0] for k in range(N)]  # x-coordinates of nodes
        Yn = [layout[k][1] for k in range(N)]  # y-coordinates
        Zn = [layout[k][2] for k in range(N)]  # z-coordinates
        Xe = []
        Ye = []
        Ze = []
        for e in Edges:
            Xe += [layout[e[0]][0], layout[e[1]][0], None]  # x-coordinates of edge ends
            Ye += [layout[e[0]][1], layout[e[1]][1], None]
            Ze += [layout[e[0]][2], layout[e[1]][2], None]

        import plotly.graph_objs as go

        trace1 = go.Scatter3d(x=Xe,
                              y=Ye,
                              z=Ze,
                              mode='lines',
                              line=dict(color='rgb(125,125,125)', width=1),
                              hoverinfo='none'
                              )

        trace2 = go.Scatter3d(x=Xn,
                              y=Yn,
                              z=Zn,
                              mode='markers',
                              name='actors',
                              marker=dict(symbol='circle',
                                          size=6,
                                          colorscale='Viridis',
                                          line=dict(color='rgb(50,50,50)', width=0.5)
                                          ),
                              # hoverinfo='text',
                              # text=['1', '2', '3', '4', '5']
                              )

        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )

        layout = go.Layout(
            title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(
                t=100
            ),
            hovermode='closest',
        )
        data = [trace1, trace2]
        fig = go.Figure(data=data, layout=layout)

        fig.show()

    def test_get_edge_positions(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        layout = graph.layout('kk', dim=3)

        N = len(layout)
        Edges = graph.get_edgelist()  # z-coordinates
        Xe = []
        Ye = []
        Ze = []
        for e in Edges:
            Xe += [layout[e[0]][0], layout[e[1]][0], None]  # x-coordinates of edge ends
            Ye += [layout[e[0]][1], layout[e[1]][1], None]
            Ze += [layout[e[0]][2], layout[e[1]][2], None]

        layout = pd.DataFrame(graph.layout('kk', dim=3).coords, columns=['x', 'y', 'z'])
        edge_positions = self.propagation_factory._get_edge_positions(graph, layout=layout)
        actual_Xe = edge_positions['x']
        actual_Ye = edge_positions['y']
        actual_Ze = edge_positions['z']

        expected_Xe = pd.Series(Xe)
        expected_Ye = pd.Series(Ye)
        expected_Ze = pd.Series(Ze)

        pd.testing.assert_series_equal(expected_Xe, actual_Xe, check_dtype=False, check_index_type=False,
                                       check_names=False)
        pd.testing.assert_series_equal(expected_Ye, actual_Ye, check_dtype=False, check_index_type=False,
                                       check_names=False)
        pd.testing.assert_series_equal(expected_Ze, actual_Ze, check_dtype=False, check_index_type=False,
                                       check_names=False)

    def test_plot_graph_vanilla(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        fig = self.propagation_factory.plot_graph(graph)
        fig.show()

    def test_plot_graph_text(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        text = [str(i) for i in range(10)]
        fig = self.propagation_factory.plot_graph(graph, text=text)
        fig.show()

    def test_plot_graph_color(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        color = [random.randint(0, 100) for _ in range(10)]
        fig = self.propagation_factory.plot_graph(graph, color=color)
        fig.show()

    def test_plot_graph_size(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        size = [random.randint(0, 100) for _ in range(10)]
        fig = self.propagation_factory.plot_graph(graph, size=size)
        fig.show()

    def test_plot_graph_symbol(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        symbol = [random.choice(['circle', 'square', 'diamond', 'cross', 'x']) for _ in range(10)]
        fig = self.propagation_factory.plot_graph(graph, symbol=symbol)
        fig.show()

    def test_get_metadata(self):
        metadata = self.propagation_factory.get_user_metadata(self.test_dataset)
        self.assertEqual(metadata.duplicated().sum(), 0)
        self.assertEqual(['username',
                          'is_usual_suspect',
                          'party',
                          'legitimacy',
                          'reputation',
                          'User type'],
                         metadata.columns.tolist())
        self.assertFalse(metadata['User type'].isna().sum())

    def test_plot_hidden_network(self):
        hidden_network = self.propagation_factory.egonet.get_hidden_network(self.test_dataset)
        fig = self.propagation_factory.plot_user_graph(hidden_network, self.test_dataset)
        fig.show()

    def test_plot_egonet(self):
        fig = self.propagation_factory.plot_egonet(self.test_dataset, self.test_user_id, 2)
        fig.show()

    def test_plot_size_over_time(self):
        fig = self.propagation_factory.plot_size_over_time(self.test_dataset, self.test_tweet_id)
        fig.show()

    def test_plot_depth_over_time(self):
        fig = self.propagation_factory.plot_depth_over_time(self.test_dataset, self.test_tweet_id)
        fig.show()

    def test_plot_max_breadth_over_time(self):
        fig = self.propagation_factory.plot_max_breadth_over_time(self.test_dataset, self.test_tweet_id)
        fig.show()

    def test_structural_virality_over_time(self):
        fig = self.propagation_factory.plot_structural_virality_over_time(self.test_dataset, self.test_tweet_id)
        fig.show()

    def _test_plot_depth_cascade_ccdf(self):
        fig = self.propagation_factory.plot_depth_cascade_ccdf(self.test_dataset)
        fig.show()

    def test_plot_size_cascade_ccdf(self):
        fig = self.propagation_factory.plot_size_cascade_ccdf(self.test_dataset)
        fig.show()

    def test_cascade_count_over_time(self):
        fig = self.propagation_factory.plot_cascade_count_over_time(self.test_dataset)
        fig.show()


if __name__ == '__main__':
    unittest.main()
