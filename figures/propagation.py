import logging
import time
from importlib.metadata import metadata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from igraph import Layout
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from pymongoarrow.schema import Schema
from sklearn import set_config

from figures.figures import MongoPlotFactory
from propagation import Egonet, NetworkMetrics, DiffusionMetrics

patch_all()

set_config(transform_output="pandas")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PropagationPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017,
                 reference_types=('quoted', 'retweeted'), layout='fruchterman_reingold',
                 threshold=0.2, delete_vertices=True, frequency='1D',
                 available_datasets=None, small_size_multiplier=15, big_size_multiplier=50,
                 user_highlight_color='rgb(0, 0, 255)', max_edges=None):
        super().__init__(host, port, available_datasets)
        self.max_edges = max_edges
        self.big_size_multiplier = big_size_multiplier
        self.small_size_multiplier = small_size_multiplier
        self.node_highlight_color = user_highlight_color
        self.layout = layout
        self.frequency = frequency
        self.delete_vertices = delete_vertices
        self.threshold = threshold
        self._layout_cache = {}
        self._backbone_cache = {}
        self.egonet = Egonet(reference_types=reference_types, host=host, port=port)

        self.network_metrics = NetworkMetrics(host=host, port=port, reference_types=reference_types,
                                              frequency=frequency)

        self.diffusion_metrics = DiffusionMetrics(egonet=self.egonet, host=host, port=port,
                                                  reference_types=reference_types)

    def plot_egonet(self, collection, author_id, depth, start_date=None, end_date=None, hashtag=None):
        egonet = self._get_egonet(collection, author_id, depth, start_date, end_date, hashtag)
        layout = self.compute_layout(egonet)
        author_id_node_index = egonet.vs.find(author_id=author_id).index

        return self.plot_user_graph(egonet, collection, layout, highlight_node_index=author_id_node_index)

    def _get_egonet(self, collection, author_id, depth, start_date, end_date, hashtag):
        egonet = self.egonet.get_egonet(collection, author_id, depth, start_date, end_date, hashtag)

        if self.max_edges is not None and egonet.ecount() > self.max_edges:
            logger.warning(f'Egonet for {author_id} has {egonet.ecount()} edges, reducing to {self.max_edges}')
            # Get author_id node just in case it gets removed in the backbone pruning
            author_id_node = egonet.vs.find(author_id=author_id)

            egonet = compute_backbone(egonet, threshold=0, delete_vertices=self.delete_vertices,
                                      max_edges=self.max_edges)
            # Make sure author_id is in the graph, add it otherwise
            try:
                egonet.vs.find(author_id=author_id).index
            except ValueError:
                author_id_data = {key: author_id_node[key] for key in author_id_node.attributes()}
                egonet.add_vertex(name=author_id, **author_id_data)

        # # Make sure all components are linked to the author_id
        author_id_index = egonet.vs.find(author_id=author_id).index
        for component in egonet.connected_components(mode='weak'):
            if author_id_index not in component:
                egonet.add_edge(author_id_index, component[0])

        return egonet

    def plot_hidden_network(self, dataset, start_date=None, end_date=None, hashtags=None):
        # if start_date, end_date or hashtag are not none we need to recompute the graph and layout
        start_date, end_date = self._validate_dates(dataset, start_date, end_date)
        if start_date or end_date or hashtags:
            hidden_network = self._compute_hidden_network_backbone(dataset, start_date, end_date, hashtags)

            layout = self.compute_layout(hidden_network)
        else:
            # Load hidden network from db
            if self.threshold > 0:
                hidden_network = self._get_backbone(dataset)
            else:
                hidden_network = self.egonet.get_hidden_network(dataset)
            layout = self._get_layout(dataset)

        return self.plot_user_graph(hidden_network, dataset, layout)

    def _get_backbone(self, dataset):
        if dataset not in self._backbone_cache:
            try:
                backbone = self._load_graph_from_mongodb(dataset, 'hidden_network_backbone')
            except Exception as e:
                logger.error(f'Error loading hidden network backbone: {e}. Recomputing')
                backbone = self._compute_hidden_network_backbone(dataset)

            self._backbone_cache[dataset] = backbone
        return self._backbone_cache[dataset]

    def _get_layout(self, dataset):
        if dataset not in self._layout_cache:
            try:
                layout = self._load_layout_from_mongodb(dataset, 'hidden_network_layout')
            except Exception as e:
                logger.error(f'Error loading hidden network layout: {e}. Recomputing')
                if self.threshold > 0:
                    hidden_network = self._get_backbone(dataset)
                else:
                    hidden_network = self.egonet.get_hidden_network(dataset)
                layout = self.compute_layout(hidden_network)
                self._persist_layout_to_mongodb(layout, dataset, 'hidden_network_layout')

            self._layout_cache[dataset] = layout
        return self._layout_cache[dataset]

    def _compute_hidden_network_backbone(self, dataset, start_date=None, end_date=None, hashtags=None):
        hidden_network = self.egonet.get_hidden_network(dataset, start_date=start_date, end_date=end_date,
                                                        hashtags=hashtags)
        if hidden_network.vcount() == 0:
            return hidden_network

        backbone = compute_backbone(hidden_network, threshold=self.threshold, delete_vertices=self.delete_vertices)

        return backbone

    def compute_layout(self, network):
        logger.debug(
            f'Computing {self.layout} layout for graph {network.vcount()} vertices and {network.ecount()} edges')
        start_time = time.time()
        layout = network.layout(self.layout, dim=3)
        layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
        logger.debug(f'Layout computed in {time.time() - start_time} seconds')
        return layout

    def plot_user_graph(self, user_graph, collection, layout, highlight_node_index=None):
        metadata = self.get_user_metadata(collection, author_ids=user_graph.vs['author_id'])
        metadata = metadata.reindex(user_graph.vs['author_id'])
        # Try to patch missing metadata with graph info if available
        if metadata.isna().any().any():
            logger.warning('Missing metadata for some nodes, trying to patch with graph info')
            missing_nodes = metadata.index[metadata.drop(columns='party').isna().any(axis=1)].to_list()
            for author_id in missing_nodes:
                try:
                    node = user_graph.vs.find(author_id=author_id)
                    metadata.loc[author_id, 'Username'] = node['username']
                    metadata.loc[author_id, 'User type'] = 'Normal'

                except ValueError:
                    logger.error(f'Node {author_id} not found in graph')
        metadata['User type'] = metadata['User type'].fillna('Unknown')

        def user_hover(x):
            username = x['username']
            user_type = x['User type']
            legitimacy = x['legitimacy_level'] if x['legitimacy_level'] else ''
            reputation = x['reputation_level'] if x['reputation_level'] else ''
            status = x['status_level'] if x['status_level'] else ''
            hover_template = (f'Username: {username}<br>User type: {user_type}<br>'
                              f'Legitimacy: {legitimacy}<br>Reputation: {reputation}<br>Status: {status}')
            return hover_template

        text = metadata.apply(user_hover, axis=1)

        size = metadata['reputation']
        # Add 1 offset and set 1 as minimum size
        size = size + 1
        size = size.fillna(1)

        color = metadata['legitimacy'].copy()

        if highlight_node_index is not None:
            color = color.to_list()
            color[highlight_node_index] = self.node_highlight_color

        # Available markers ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
        marker_map = {'Normal': 'circle', 'Suspect': 'cross', 'Politician': 'diamond', 'Suspect politician':
            'square', 'Unknown': 'x'}
        symbol = metadata.apply(lambda x: marker_map[x['User type']], axis=1)

        return self.plot_graph(user_graph, layout=layout, text=text, size=size, color=color, symbol=symbol)

    def plot_propagation_tree(self, dataset, tweet_id):
        try:
            propagation_tree = self.diffusion_metrics.get_propagation_tree(dataset, tweet_id)
        except Exception as e:
            logger.error(f'Error getting propagation tree: {e}. Recomputing')
            propagation_tree = self.diffusion_metrics.compute_propagation_tree(dataset, tweet_id)

        if self.max_edges is not None and propagation_tree.ecount() > self.max_edges:
            # sample edges
            edges = np.random.choice(propagation_tree.ecount(), self.max_edges, replace=False)
            propagation_tree = propagation_tree.subgraph_edges(edges)
        try:
            original_node_index = propagation_tree.vs.find(tweet_id=tweet_id).index
        except ValueError:
            logger.error(f'Tweet {tweet_id} not found in propagation tree')
            original_node_index = None

        layout = self.compute_layout(propagation_tree)
        return self.plot_user_graph(propagation_tree, dataset, layout, highlight_node_index=original_node_index)

    def plot_size_over_time(self, dataset, tweet_id):
        try:
            size_over_time = self.diffusion_metrics.get_size_over_time(dataset, tweet_id)
        except Exception as e:
            logger.error(f'Error getting size over time: {e}. Recomputing')
            graph = self.diffusion_metrics.compute_propagation_tree(dataset, tweet_id)
            size_over_time = self.diffusion_metrics.compute_size_over_time(graph)
        return plot_time_series(size_over_time, 'Size over time', 'Time', 'Size')

    def plot_depth_over_time(self, dataset, tweet_id):
        try:
            depth_over_time = self.diffusion_metrics.get_depth_over_time(dataset, tweet_id)
        except Exception as e:
            logger.error(f'Error getting depth over time: {e}. Recomputing')
            graph = self.diffusion_metrics.compute_propagation_tree(dataset, tweet_id)
            shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
            depth_over_time = self.diffusion_metrics.compute_depth_over_time(shortest_paths=shortest_paths)
            # remove all input edges whose input edges link to the original tweet
            graph.delete_edges(graph.es.select(lambda e: e.source == graph.vs.find(tweet_id=tweet_id).index))
        return plot_time_series(depth_over_time, 'Depth over time', 'Time', 'Depth')

    def plot_max_breadth_over_time(self, dataset, tweet_id):
        try:
            max_breadth_over_time = self.diffusion_metrics.get_max_breadth_over_time(dataset, tweet_id)
        except Exception as e:
            logger.error(f'Error getting max breadth over time: {e}. Recomputing')
            graph = self.diffusion_metrics.compute_propagation_tree(dataset, tweet_id)
            shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
            max_breadth_over_time = self.diffusion_metrics.compute_max_breadth_over_time(shortest_paths)
        return plot_time_series(max_breadth_over_time, 'Max breadth over time', 'Time', 'Max breadth')

    def plot_structural_virality_over_time(self, dataset, tweet_id):
        try:
            structural_virality_over_time = self.diffusion_metrics.get_structural_virality_over_time(dataset, tweet_id)
        except Exception as e:
            logger.error(f'Error getting structural virality over time: {e}. Recomputing')
            graph = self.diffusion_metrics.compute_propagation_tree(dataset, tweet_id)
            structural_virality_over_time = self.diffusion_metrics.compute_structural_virality_over_time(graph)
        return plot_time_series(structural_virality_over_time, 'Structural virality over time', 'Time',
                                'Structural virality')

    def plot_size_cascade_ccdf(self, dataset):
        try:
            size_cascade = self.diffusion_metrics.get_size_cascade_ccdf(dataset)
        except Exception as e:
            logger.error(f'Error getting size cascade CCDF: {e}. Recomputing')
            size_cascade = self.diffusion_metrics.compute_size_cascade_ccdf(dataset)
        bad_columns = size_cascade.columns[(~size_cascade.isna()).sum() <= 1].to_list()
        size_cascade = size_cascade.drop(columns=bad_columns)
        # Interpolate by columns to fill the nan gaps
        size_cascade = size_cascade.interpolate(method='slinear', axis=0)
        return plot_time_series(size_cascade, 'Size cascade CCDF', 'Size', 'CCDF')

    def plot_cascade_count_over_time(self, dataset):
        try:
            cascade_count_over_time = self.diffusion_metrics.get_cascade_count_over_time(dataset)
        except Exception as e:
            logger.error(f'Error getting cascade count over time: {e}. Recomputing')
            cascade_count_over_time = self.diffusion_metrics.compute_cascade_count_over_time(dataset)
        return plot_time_series(cascade_count_over_time, 'Cascade count over time', 'Time', 'Cascade count')

    def get_user_metadata(self, dataset, author_ids=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)

        pipeline = [
            {'$project': {'_id': 0,
                          'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party'
                          }},
            {'$group': {'_id': '$author_id',
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'},
                        }},
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]},

                          }}

        ]

        if author_ids:
            pipeline.insert(0, {'$match': {'author.id': {'$in': author_ids}}})
        schema = Schema({'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str})
        metadata = database.get_collection('raw').aggregate_pandas_all(pipeline, schema=schema)
        network_metrics_pipeline = [
            {'$project': {'_id': 0,
                          'author_id': 1,
                          'legitimacy': 1,
                          'reputation': '$average_reputation',
                          'status': '$average_status',
                          'legitimacy_level': 1,
                          'reputation_level': 1,
                          'status_level': 1, }
             },
        ]
        network_metrics_schema = Schema({'author_id': str, 'legitimacy': float,
                                         'reputation': float, 'status': float,
                                         'legitimacy_level': str, 'reputation_level': str, 'status_level': str})
        network_metrics = database.get_collection('network_metrics').aggregate_pandas_all(network_metrics_pipeline,
                                                                                          schema=network_metrics_schema)
        client.close()
        metadata = metadata.set_index('author_id')
        metadata = metadata.join(network_metrics.set_index('author_id'))
        metadata['User type'] = metadata.apply(transform_user_type, axis=1).fillna('Unknown')

        return metadata

    def plot_graph(self, graph, layout, symbol=None, size=None, color=None, text=None, normalize=True):
        if normalize:
            # apply logarithm on color and size if available
            if color is not None:
                color = pd.Series(color)
                color = color.apply(lambda x: np.log(x + 1) if isinstance(x, (int, float)) else x)

            if size is not None:
                original_size = size
                size = np.log(size + 1) / np.log(size.max() + 1) * self.small_size_multiplier

        if isinstance(layout, Layout):
            layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])

        if layout.shape[0] != graph.vcount():
            logger.warning('Layout and graph have different number of vertices. Make sure persisted layout'
                           ' is correct. Recomputing layout')
            layout = self.compute_layout(graph)
            layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
        logger.debug('Computing plot for network')
        logger.debug(graph.summary())
        start_time = time.time()
        edge_positions = get_edge_positions(graph, layout)

        edge_trace = go.Scatter3d(x=edge_positions['x'],
                                  y=edge_positions['y'],
                                  z=edge_positions['z'],
                                  mode='lines',
                                  line=dict(color='rgb(125,125,125)', width=1),
                                  hoverinfo='none',
                                  name='Interactions',
                                  showlegend=False
                                  )

        node_trace = go.Scatter3d(x=layout['x'],
                                  y=layout['y'],
                                  z=layout['z'],
                                  mode='markers',
                                  marker=dict(symbol=symbol,
                                              size=size if size is not None else 3,
                                              color=color if color is not None else 'rgb(255, 234, 208)',
                                              # coloscale set to $champagne: #ffead0ff;
                                              # to $bright-pink-crayola: #f76f8eff;
                                              colorscale=[[0, 'rgb(255, 234, 208)'], [1, 'rgb(247, 111, 142)']],
                                              colorbar=dict(thickness=20, title='Legitimacy'),
                                              line=dict(color='rgb(50,50,50)', width=0.5),
                                              ),
                                  text=text,
                                  hovertemplate='%{text}' if text is not None else None,
                                  name='',
                                  )

        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showspikes=False,
                    title=''
                    )

        layout = go.Layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(
                t=0,
                l=0,
                r=0,
                b=0
            ),
            hovermode='closest',
            autosize=True,
        )

        data = [edge_trace, node_trace]
        fig = go.Figure(data=data, layout=layout)

        if size is not None:
            # Create a custom legend for marker sizes
            if normalize:
                legend_values = np.linspace(original_size.min(), original_size.max(), 5)  #
                legend_values = np.array(sorted(set(legend_values.astype(int))))
                legend_sizes = np.log(legend_values + 1) / np.log(original_size.max() + 1) * self.small_size_multiplier
            else:
                legend_values = np.linspace(size.min(), size.max(), 5)
                legend_sizes = legend_values * self.small_size_multiplier
            for size, value in zip(legend_sizes, legend_values):
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color='lightgrey'
                    ),
                    legendgroup='size',
                    showlegend=True,
                    name=f'{value}',
                    # remove background
                    hoverinfo='skip',

                ))

            # Update layout to position the legends
            fig.update_layout(
                legend=dict(
                    x=1.1,
                    y=0.925,
                    traceorder='normal',
                    xanchor='left',
                    yanchor='top',
                    itemclick=False,  # Disable click events on legend items,
                    title='Reputation',
                ),
                scene=dict(
                    xaxis=dict(axis),
                    yaxis=dict(axis),
                    zaxis=dict(axis),
                ),
                xaxis={'showgrid': False, 'zeroline': False, 'showline': False, 'showticklabels': False},
                yaxis={'showgrid': False, 'zeroline': False, 'showline': False, 'showticklabels': False}

            )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.75, y=0.75, z=0.75),
        )
        fig.update_layout(scene_camera=camera)
        logger.debug(f'Graph plot computed in {time.time() - start_time} seconds')
        return fig

    def get_cascade_size(self, dataset, tweet_id):
        return self.diffusion_metrics.get_cascade_size(dataset, tweet_id)

    def persist(self, datasets):
        # Save layouts and backbone if any
        for dataset in datasets:
            if self.threshold > 0:
                logger.info(f'Persisting hidden network layout and backbone for dataset {dataset}')
                hidden_network = self._compute_hidden_network_backbone(dataset)
                self._persist_graph_to_mongodb(hidden_network, dataset, 'hidden_network_backbone')
            else:
                logger.info(f'Persisting hidden network layout for dataset {dataset}')
                hidden_network = self.egonet.get_hidden_network(dataset)

            layout = self.compute_layout(hidden_network)
            if not layout.empty:
                self._persist_layout_to_mongodb(layout, dataset, 'hidden_network_layout')
            else:
                logger.error(f'Empty layout for dataset {dataset}')


    def _persist_layout_to_mongodb(self, layout, dataset, collection_name):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection(collection_name)
        collection.drop()
        collection.insert_many(layout.to_dict(orient='records'))
        client.close()

    def _load_layout_from_mongodb(self, dataset, collection_name):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        # Check collection actually exists
        if collection_name not in database.list_collection_names():
            logger.error(f'Layout collection {collection_name} not found for dataset {dataset}. Is it persisted?')
            raise ValueError(f'Layout collection {collection_name} not found for dataset {dataset}')
        collection = database.get_collection(collection_name)
        layout = collection.aggregate_pandas_all([{'$project': {'_id': 0}}])
        client.close()
        return layout


def compute_alphas(graph):
    # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
    weights = np.array(graph.es['weight_norm'])
    degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
    alphas = (1 - weights) ** (degrees - 1)
    return pd.Series(alphas)


def compute_backbone(graph, threshold=0.05, delete_vertices=True, max_edges=None):
    alphas = compute_alphas(graph)
    rank = alphas.argsort()
    alphas = alphas[rank]
    good = rank[alphas > threshold]

    if max_edges:
        good = good[:max_edges]
    logger.debug(f'Pruning {graph.ecount() - len(good)}  edges from {graph.ecount()}: {len(good)} edges remaining')
    backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)

    return backbone


def get_edge_positions(graph, layout):
    # edges = pd.DataFrame(graph.get_edgelist(), columns=['source', 'target'])
    # edge_positions = layout.iloc[edges.values.flatten()].reset_index(drop=True)
    # nones = edge_positions[2::3].assign(x=None, y=None, z=None)
    # edge_positions = pd.concat([edge_positions, nones]).sort_index().reset_index(drop=True)
    if isinstance(layout, Layout):
        layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
    layout = list(layout.to_records(index=False))
    x = []
    y = []
    z = []
    for edges in graph.get_edgelist():
        x += [layout[edges[0]][0], layout[edges[1]][0], None]  # x-coordinates of edge ends
        y += [layout[edges[0]][1], layout[edges[1]][1], None]
        z += [layout[edges[0]][2], layout[edges[1]][2], None]

    edge_positions = pd.DataFrame({'x': x, 'y': y, 'z': z})
    return edge_positions


def plot_time_series(data, title, x_label, y_label):
    if isinstance(data, pd.Series):
        data = data.to_frame()
    fig = px.line(data, x=data.index, y=data.columns)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    # fig.update_layout(title=title)
    return fig


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
