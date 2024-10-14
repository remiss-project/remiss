import itertools
import random
import unittest
from unittest.mock import Mock

import igraph as ig
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all

from figures.propagation import PropagationPlotFactory, compute_backbone, plot_time_series, get_edge_positions
from tests.conftest import create_test_data_from_edges

patch_all()


class PropagationFactoryTestCase(unittest.TestCase):
    def setUp(self):
        self.propagation_factory = PropagationPlotFactory(available_datasets=['test_dataset_2'], preload=False)
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.test_user_id = '2201623465'
        self.test_tweet_id = '1167074391315890176'

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

        # fig.show()

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
        edge_positions = get_edge_positions(graph, layout=layout)
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
        fig = self.propagation_factory.plot_graph(graph, layout=graph.layout('kk', dim=3))
        # fig.show()

    def test_plot_graph_text(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        text = [str(i) for i in range(10)]
        layout = graph.layout('kk', dim=3)
        fig = self.propagation_factory.plot_graph(graph, text=text, layout=layout)
        # fig.show()

    def test_plot_graph_color(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        color = [random.randint(0, 100) for _ in range(10)]
        layout = graph.layout('kk', dim=3)
        fig = self.propagation_factory.plot_graph(graph, color=color, layout=layout)
        # fig.show()

    def test_plot_graph_size(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        size = np.array([random.randint(0, 100) for _ in range(10)])
        layout = graph.layout('kk', dim=3)
        fig = self.propagation_factory.plot_graph(graph, size=size, layout=layout)
        # fig.show()

    def test_plot_graph_symbol(self):
        graph = ig.Graph()
        graph.add_vertices(10)
        graph.add_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)])
        symbol = [random.choice(['circle', 'square', 'diamond', 'cross', 'x']) for _ in range(10)]
        layout = graph.layout('kk', dim=3)
        fig = self.propagation_factory.plot_graph(graph, symbol=symbol, layout=layout)
        # fig.show()

    def test_get_metadata(self):
        metadata = self.propagation_factory.get_user_metadata(self.test_dataset)
        self.assertEqual(metadata.duplicated().sum(), 0)
        self.assertEqual(['username',
                          'is_usual_suspect',
                          'party',
                          'legitimacy',
                          'reputation',
                          'status',
                          'legitimacy_level',
                          'reputation_level',
                          'status_level',
                          'User type'],
                         metadata.columns.tolist())
        self.assertFalse(metadata['User type'].isna().sum())


    def test_plot_hidden_network_2(self):
        fig = self.propagation_factory.plot_hidden_network(self.test_dataset)
        # fig.show()



    def test_plot_egonet(self):
        fig = self.propagation_factory.plot_egonet(self.test_dataset, self.test_user_id, 2)
        fig.show()

    def test_plot_egonet_max_edges(self):
        self.propagation_factory.max_edges_propagation_tree = 10
        fig = self.propagation_factory.plot_egonet(self.test_dataset, self.test_user_id, 2)
        fig.show()

    def test_plot_egonet_missing(self):
        with self.assertRaises(ValueError):
            self.propagation_factory.plot_egonet(self.test_dataset, 'potato', 2)

    def test_hidden_network_backbone(self):
        self.propagation_factory.threshold = 0.95
        fig = self.propagation_factory.plot_hidden_network(self.test_dataset)

        pruned = len(fig.data[1].x)

        self.propagation_factory._hidden_network_plot_cache = {}
        self.propagation_factory._layout_cache = {}
        self.propagation_factory.threshold = 0
        fig = self.propagation_factory.plot_hidden_network(self.test_dataset)

        full =  len(fig.data[1].x)
        self.assertGreater(full, pruned)

    def test_plot_propagation_tree(self):
        fig = self.propagation_factory.plot_propagation_tree(self.test_dataset, self.test_tweet_id)
        fig.show()

    def test_plot_propagation_tree_2(self):
        test_twett = '1167074391315890176'
        fig = self.propagation_factory.plot_propagation_tree(self.test_dataset, test_twett)
        fig.show()

    def test_plot_size_over_time(self):
        fig = self.propagation_factory.plot_size_over_time(self.test_dataset, self.test_tweet_id)
        # fig.show()

    def test_plot_depth_over_time(self):
        fig = self.propagation_factory.plot_depth_over_time(self.test_dataset, self.test_tweet_id)
        # fig.show()

    def test_plot_max_breadth_over_time(self):
        fig = self.propagation_factory.plot_max_breadth_over_time(self.test_dataset, self.test_tweet_id)
        # fig.show()

    def test_structural_virality_over_time(self):
        fig = self.propagation_factory.plot_structural_virality_over_time(self.test_dataset, self.test_tweet_id)
        # fig.show()

    def test_plot_size_cascade_ccdf(self):
        fig = self.propagation_factory.plot_size_cascade_ccdf(self.test_dataset)
        fig.show()

    @unittest.skip('Remote test')
    def test_plot_size_cascade_ccdf_remote(self):
        self.propagation_factory.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'

        for dataset in ['Openarms', 'MENA_Agressions', 'MENA_Ajudes', 'Barcelona_2019', 'Generalitat_2021',
                        'Andalucia_2022', 'Generales_2019']:
            size = self.propagation_factory.diffusion_metrics.get_size_cascade_ccdf(dataset)
            # Drop columns with only a single non-nan value
            bad_columns = size.columns[(~size.isna()).sum() <= 1].to_list()
            size = size.drop(columns=bad_columns)
            # Interpolate by columns to fill the nan gaps
            size = size.interpolate(method='slinear', axis=0)
            fig = plot_time_series(size, f'Size cascade CCDF {dataset}', 'Size', 'CCDF')
            fig.show()

    def test_cascade_count_over_time(self):
        fig = self.propagation_factory.plot_cascade_count_over_time(self.test_dataset)
        # fig.show()

    def test_legend(self):
        import plotly.graph_objects as go
        import plotly.express as px

        # Sample data
        df = px.data.iris()

        # Create the main scatter plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['sepal_width'],
            y=df['sepal_length'],
            mode='markers',
            marker=dict(
                size=df['petal_length'],
                color=df['species'].astype('category').cat.codes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Species")
            ),
            text=df['species'],
            name='Data points'
        ))

        # Create a custom legend for marker sizes
        sizes = [5, 10, 15, 20, 25]  # Example sizes for the legend
        for size in sizes:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=size,
                    color='lightgrey'
                ),
                legendgroup='size',
                showlegend=True,
                name=f'Size {size}'
            ))

        # Update layout to position the legends
        fig.update_layout(
            title='Scatter plot with color and size',
            xaxis_title='Sepal Width',
            yaxis_title='Sepal Length',
            legend=dict(
                x=1.05,
                y=1,
                traceorder='normal',
                xanchor='left',
                yanchor='top',
                itemclick=False  # Disable click events on legend items
            )
        )

        # fig.show()

    def test_backbone(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '2', '1', '1', '1', '1', '1'],
                                       'target': ['2', '3', '3', '2', '2', '2', '2', '4']})
        test_data = create_test_data_from_edges(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        network = self.propagation_factory.egonet._compute_hidden_network(self.tmp_dataset)
        backbone = compute_backbone(network, threshold=0.4)
        actual = {frozenset(x) for x in backbone.get_edgelist()}
        expected = {frozenset((0, 1))}
        self.assertEqual(expected, actual)

    def test_backbone_2(self):
        # Create a test graph
        alpha = 0.05

        network = ig.Graph.Erdos_Renyi(250, 0.02, directed=False)
        network.es["weight_norm"] = np.random.uniform(0.1, 1, network.ecount())

        # Test the backbone filter
        backbone = compute_backbone(network, threshold=alpha)

        # Ensure that all edge weights are below alpha
        for edge in backbone.get_edgelist():
            weight = backbone.es[backbone.get_eid(*edge)]['weight_norm']
            degree = backbone.degree(edge[0])
            edge_alpha = (1 - weight) ** (degree - 1)
            self.assertGreater(edge_alpha, alpha)

    def test_backbone_3(self):
        # Create a test graph
        max_edges = 100

        network = ig.Graph.Erdos_Renyi(250, 0.02, directed=False)
        network.es["weight_norm"] = np.random.uniform(0, 0.5, network.ecount())

        # Test the backbone filter
        backbone = compute_backbone(network, max_edges=max_edges)

        # Ensure that all edge weights are below alpha
        backbone_alphas = {}
        for edge in backbone.get_edgelist():
            weight = backbone.es[backbone.get_eid(*edge)]['weight_norm']
            degree = backbone.degree(edge[0])
            edge_alpha = (1 - weight) ** (degree - 1)
            backbone_alphas[edge] = edge_alpha
        backbone_alphas = pd.Series(backbone_alphas, name='alpha')

        network_alphas = {}
        for edge in network.get_edgelist():
            weight = network.es[network.get_eid(*edge)]['weight_norm']
            degree = network.degree(edge[0])
            edge_alpha = (1 - weight) ** (degree - 1)
            network_alphas[edge] = edge_alpha

        network_alphas = pd.Series(network_alphas, name='alpha')
        network_alphas = network_alphas.sort_values(ascending=False)

        backbone_alphas = backbone_alphas.sort_values(ascending=False)
        self.assertEqual(max_edges, len(backbone_alphas))

    def test_backbone_full(self):
        self.propagation_factory.threshold = 0.95
        backbone = self.propagation_factory._compute_hidden_network_backbone(self.test_dataset)
        self.assertEqual(156, backbone.vcount())
        self.assertEqual(80, backbone.ecount())

    def test_backbone_full_nothing(self):
        self.propagation_factory.threshold = 1
        backbone = self.propagation_factory._compute_hidden_network_backbone(self.test_dataset)
        self.assertEqual(backbone.vcount(), 0)
        self.assertEqual(backbone.ecount(), 0)

    @unittest.skip('Remote test')
    def test_depth_over_time_2(self):
        test_tweet = '1182192005377601536'
        self.propagation_factory.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        self.propagation_factory.diffusion_metrics.egonet.host = 'mongodb://srvinv02.esade.es'

        fig = self.propagation_factory.plot_depth_over_time('Openarms', test_tweet)
        fig.show()

    def test_depth_over_time_3(self):
        test_tweet = '1167074391315890176'

        fig = self.propagation_factory.plot_depth_over_time(self.test_dataset, test_tweet)
        fig.show()

    @unittest.skip('Remote test')
    def test_depth_over_time_4(self):
        test_tweet = '1167100545800318976'
        self.propagation_factory.host = 'mongodb://srvinv02.esade.es'
        self.propagation_factory.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        self.propagation_factory.diffusion_metrics.egonet.host = 'mongodb://srvinv02.esade.es'

        tree = self.propagation_factory.diffusion_metrics.compute_propagation_tree('Openarms', test_tweet)
        print(tree)

    def test_plot_propagation_generation(self):
        class MockModel:
            def __init__(self, limit=1):
                self.current = 0
                self.limit = limit

            def predict(self, X):
                pred = np.zeros(X.shape[0], dtype=int)
                remaining = self.limit - self.current
                if remaining > 0:
                    available = np.minimum(remaining, X.shape[0])
                    pred[:available] = 1
                    self.current += available
                    np.random.shuffle(pred)
                return pred

        model = MockModel(limit=50)
        self.propagation_factory._load_propagation_model = Mock(return_value=model)
        fig = self.propagation_factory.plot_propagation_generation(self.test_dataset, '1167074391315890176')
        fig.show()

    def test_sizes_and_markers(self):
        legitimacy = [1, 10, 100]
        reputation_levels = ['Low', 'Medium', 'High']
        status_levels = ['Low', 'Medium', 'High']
        marker_map = {'Normal': 'circle', 'Suspect': 'cross', 'Politician': 'diamond', 'Suspect politician':
            'square', 'Unknown': 'x'}
        user_types = list(marker_map.keys())
        combinations = itertools.product(legitimacy, reputation_levels, status_levels, user_types)

        metadata = pd.DataFrame(combinations, columns=['legitimacy', 'reputation_level', 'status_level', 'User type'])
        metadata['legitimacy_level'] = metadata['legitimacy'].map({1: 'Low', 10: 'Medium', 100: 'High'})
        metadata['username'] = metadata.index
        metadata['author_id'] = metadata.index
        # get random graph of size of metadata rows
        graph = ig.Graph.Tree(metadata.shape[0], 2)
        graph.vs['name'] = metadata['username']
        graph.vs['author_id'] = metadata['author_id']
        graph.vs['legitimacy'] = metadata['legitimacy']
        graph.vs['reputation_level'] = metadata['reputation_level']
        graph.vs['status_level'] = metadata['status_level']
        graph.vs['User type'] = metadata['User type']
        layout = graph.layout('kk', dim=3)

        self.propagation_factory.get_user_metadata = Mock(return_value=metadata)
        fig = self.propagation_factory.plot_user_graph(graph, self.test_dataset, layout)
        fig.show()

        self.propagation_factory.plot_graph = Mock(return_value=fig)
        call_arguments = self.propagation_factory.plot_graph.call_args_list
        self.propagation_factory.plot_user_graph(graph, self.test_dataset, layout)
        for call in call_arguments:
            args, kwargs = call
            graph = args[0]
            color = kwargs.get('color')
            size = kwargs.get('size')

            self.assertTrue((color == metadata['legitimacy_level'].map(self.propagation_factory.colors)).all())
            self.assertTrue((size == metadata['reputation_level'].map(self.propagation_factory.sizes)).all())
            self.assertIsInstance(graph, ig.Graph)

    def test_cascade_count_over_time_barcelona19(self):
        self.propagation_factory.host = 'mongodb://srvinv02.esade.es'
        self.propagation_factory.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        self.propagation_factory.diffusion_metrics.egonet.host = 'mongodb://srvinv02.esade.es'

        fig = self.propagation_factory.plot_cascade_count_over_time('Andalucia_2022')
        fig.show()


if __name__ == '__main__':
    unittest.main()
