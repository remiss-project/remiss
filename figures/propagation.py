import datetime
import logging
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymongo
import pymongoarrow
from igraph import Layout
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from sklearn import set_config
from tqdm import tqdm

from figures.figures import MongoPlotFactory
from propagation import Egonet, NetworkMetrics, DiffusionMetrics

pymongoarrow.monkey.patch_all()

set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


class PropagationPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017,
                 reference_types=('quoted', 'retweeted'), layout='fruchterman_reingold',
                 threshold=0.2, delete_vertices=True, frequency='1D',
                 available_datasets=None, small_size_multiplier=10, big_size_multiplier=50,
                 user_highlight_color='rgb(0, 0, 255)', load_from_mongodb=True):
        super().__init__(host, port, available_datasets)
        self.big_size_multiplier = big_size_multiplier
        self.small_size_multiplier = small_size_multiplier
        self.user_highlight_color = user_highlight_color
        self.layout = layout
        self.frequency = frequency
        self.egonet = Egonet(reference_types=reference_types, host=host, port=port,
                             threshold=threshold, delete_vertices=delete_vertices)

        self.network_metrics = NetworkMetrics(host=host, port=port, reference_types=reference_types,
                                              frequency=frequency)

        self.diffusion_metrics = DiffusionMetrics(host=host, port=port, reference_types=reference_types)
        self._hidden_network_layouts = {}

        if load_from_mongodb:
            try:
                self.network_metrics.load_from_mongodb(self.available_datasets)
            except Exception as ex:
                logger.error(f'Error loading network metrics with error {ex}')

            try:
                self.egonet.load_from_mongodb(self.available_datasets)
            except Exception as ex:
                logger.error(f'Error loading egonet with error {ex}')

            try:
                self.diffusion_metrics.load_from_mongodb(self.available_datasets)
            except Exception as ex:
                logger.error(f'Error loading diffusion metrics with error {ex}')

            try:
                self.load_from_mongodb(self.available_datasets)
            except Exception as ex:
                logger.error(f'Error loading hidden network layout with error {ex}')

    def plot_egonet(self, collection, author_id, depth, start_date=None, end_date=None, hashtag=None):
        network = self.egonet.get_egonet(collection, author_id, depth, start_date, end_date, hashtag)
        layout = self.compute_layout(network)
        return self.plot_user_graph(network, collection, layout, author_id=author_id)

    def plot_hidden_network(self, collection, author_id=None, start_date=None, end_date=None, hashtag=None):
        if self.egonet.threshold > 0:
            hidden_network = self.egonet.get_hidden_network_backbone(collection, start_date, end_date, hashtag)
        else:
            hidden_network = self.egonet.get_hidden_network(collection, start_date, end_date, hashtag)
        layout = self.get_hidden_network_layout(hidden_network, collection, start_date, end_date, hashtag)
        return self.plot_graph(hidden_network, layout=layout, author_id=author_id)

    def get_hidden_network_layout(self, hidden_network, collection, start_date=None, end_date=None, hashtags=None):
        # if start_date, end_date or hashtag are not none we need to recompute the layout
        start_date, end_date = self._validate_dates(collection, start_date, end_date)
        if start_date or end_date or hashtags:
            if self.egonet.threshold > 0:
                hidden_network = self.egonet.get_hidden_network_backbone(collection, start_date, end_date, hashtags)
            else:
                hidden_network = self.egonet.get_hidden_network(collection, start_date, end_date, hashtags)
            layout = self.compute_layout(hidden_network)
            layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
            return layout
        if collection not in self._hidden_network_layouts:
            layout = self.compute_layout(hidden_network)
            layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
            self._hidden_network_layouts[collection] = layout
        return self._hidden_network_layouts[collection]

    def _validate_dates(self, dataset, start_date, end_date):
        dataset_start_date, dataset_end_date = self.get_date_range(dataset)
        if start_date:
            start_date = pd.to_datetime(start_date).date()
        if end_date:
            end_date = pd.to_datetime(end_date).date()

        if start_date == dataset_start_date:
            start_date = None
        if end_date == dataset_end_date:
            end_date = None

        return start_date, end_date

    def compute_layout(self, network):
        logger.info(f'Computing {self.layout} layout')
        start_time = time.time()
        layout = network.layout(self.layout, dim=3)
        logger.info(f'Layout computed in {time.time() - start_time} seconds')
        return layout

    def plot_user_graph(self, user_graph, collection, layout=None, author_id=None):
        metadata = self.get_user_metadata(collection)
        metadata = metadata.reindex(user_graph.vs['author_id'])
        metadata['User type'] = metadata['User type'].fillna('Unknown')

        def format_user_string(x):
            username = x['username']
            user_type = x['User type']
            return f'Username: {username}<br>' \
                   f'User type: {user_type}'

        text = metadata.apply(format_user_string, axis=1)

        size = metadata['reputation']
        # Add 1 offset and set 1 as minimum size
        size = size + 1
        size = size.fillna(1)
        if len(user_graph.vs) > 100:
            size = size / size.max() * self.small_size_multiplier
        else:
            size = size / size.max() * self.big_size_multiplier

        color = metadata['legitimacy']

        if author_id is not None:
            color[user_graph.vs.find(author_id=author_id).index] = self.user_highlight_color

        # Available markers ['circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x']
        marker_map = {'Normal': 'circle', 'Suspect': 'cross', 'Politician': 'diamond', 'Suspect politician':
            'square', 'Unknown': 'x'}
        symbol = metadata.apply(lambda x: marker_map[x['User type']], axis=1)
        # layout = self.get_hidden_network_layout(collection)

        return self.plot_graph(user_graph, layout=layout, text=text, size=size, color=color, symbol=symbol)

    def plot_propagation_tree(self, dataset, tweet_id):
        propagation_tree = self.diffusion_metrics.get_propagation_tree(dataset, tweet_id)
        return self.plot_graph(propagation_tree)

    def plot_size_over_time(self, dataset, tweet_id):
        size_over_time = self.diffusion_metrics.get_size_over_time(dataset, tweet_id)
        return self.plot_time_series(size_over_time, 'Size over time', 'Time', 'Size')

    def plot_depth_over_time(self, dataset, tweet_id):
        depth_over_time = self.diffusion_metrics.get_depth_over_time(dataset, tweet_id)
        return self.plot_time_series(depth_over_time, 'Depth over time', 'Time', 'Depth')

    def plot_max_breadth_over_time(self, dataset, tweet_id):
        max_breadth_over_time = self.diffusion_metrics.get_max_breadth_over_time(dataset, tweet_id)
        return self.plot_time_series(max_breadth_over_time, 'Max breadth over time', 'Time', 'Max breadth')

    def plot_structural_virality_over_time(self, dataset, tweet_id):
        structural_virality_over_time = self.diffusion_metrics.get_structural_virality_over_time(dataset, tweet_id)
        return self.plot_time_series(structural_virality_over_time, 'Structural virality over time', 'Time',
                                     'Structural virality')

    def plot_depth_cascade_ccdf(self, dataset):
        depth_cascade = self.diffusion_metrics.get_depth_cascade_ccdf(dataset)
        return self.plot_time_series(depth_cascade, 'Depth cascade CCDF', 'Depth', 'CCDF')

    def plot_size_cascade_ccdf(self, dataset):
        size_cascade = self.diffusion_metrics.get_size_cascade_ccdf(dataset)
        return self.plot_time_series(size_cascade, 'Size cascade CCDF', 'Size', 'CCDF')

    def plot_cascade_count_over_time(self, dataset):
        cascade_count_over_time = self.diffusion_metrics.get_cascade_count_over_time(dataset)
        return self.plot_time_series(cascade_count_over_time, 'Cascade count over time', 'Time', 'Cascade count')

    def get_user_metadata(self, dataset, author_ids=None):
        metadata = self.get_verificat_user_metadata(dataset, author_ids)
        legitimacy = self.network_metrics.get_legitimacy(dataset)
        metadata = metadata.merge(legitimacy, right_index=True, left_index=True, how='left')
        reputation = self.network_metrics.get_reputation(dataset)
        reputation_mean = reputation.mean(axis=1)
        reputation_mean.name = 'reputation'
        metadata = metadata.merge(reputation_mean, right_index=True, left_index=True, how='left')
        metadata['User type'] = metadata.apply(transform_user_type, axis=1).fillna('Unknown')

        return metadata

    def get_verificat_user_metadata(self, dataset, author_ids=None):
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
                        }
             },
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]}
                          }}

        ]
        if author_ids:
            pipeline.insert(0, {'$match': {'author.id': {'$in': author_ids}}})
        schema = Schema({'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str})
        metadata = database.get_collection('raw').aggregate_pandas_all(pipeline, schema=schema)
        client.close()
        metadata = metadata.set_index('author_id')
        return metadata

    def plot_graph(self, graph, layout=None, symbol=None, size=None, color=None, text=None, author_id=None):
        if layout is None:
            if 'layout' not in graph.attributes():
                layout = graph.layout(self.layout, dim=3)

            else:
                layout = graph['layout']

        if isinstance(layout, Layout):
            layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])

        if layout.shape[0] != graph.vcount():
            logger.warning('Layout and graph have different number of vertices. Make sure persisted layout'
                           ' is correct. Recomputing layout')
            layout = self.compute_layout(graph)
            layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
        logger.info('Computing plot for network')
        logger.info(graph.summary())
        start_time = time.time()
        edge_positions = self._get_edge_positions(graph, layout)

        if author_id is not None:
            try:
                author_index = graph.vs.find(author_id=author_id).index
                if color is None:
                    color = ['rgb(255, 234, 208)'] * graph.vcount()

                color[author_index] = self.user_highlight_color
            except ValueError:
                logger.warning(f'Author id {author_id} not found in graph, not highlighting user')

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
                    title=''
                    )

        layout = go.Layout(
            showlegend=True,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            # margin=dict(
            #     t=100
            # ),
            hovermode='closest',

        )

        data = [edge_trace, node_trace]
        fig = go.Figure(data=data, layout=layout)

        # Create a custom legend for marker sizes
        sizes = [1, 2, 3, 5, 10, 15, 25]  # Example sizes for the legend
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
                name=f'{size}'
            ))

        # Update layout to position the legends
        fig.update_layout(
            legend=dict(
                x=1.05,
                y=0.995,
                traceorder='normal',
                xanchor='left',
                yanchor='top',
                itemclick=False,  # Disable click events on legend items,
                title='Reputation',
            )
        )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        fig.update_layout(scene_camera=camera)
        logger.info(f'Plot computed in {time.time() - start_time} seconds')
        return fig

    def _get_edge_positions(self, graph, layout):
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

    def plot_time_series(self, data, title, x_label, y_label):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        fig = px.line(data, x=data.index, y=data.columns)
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)
        fig.update_layout(title=title)
        return fig

    def persist(self, datasets):
        # Save layouts to mongodb
        for dataset in datasets:
            if self.egonet.threshold > 0:
                hidden_network = self.egonet.get_hidden_network_backbone(dataset)
            else:
                hidden_network = self.egonet.get_hidden_network(dataset)
            layout = self.get_hidden_network_layout(hidden_network, dataset)
            self._persist_layout_to_mongodb(layout, dataset, 'hidden_network_layout')

    def _persist_layout_to_mongodb(self, layout, dataset, collection_name):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection(collection_name)
        collection.drop()
        collection.insert_many(layout.to_dict(orient='records'))
        client.close()

    def load_from_mongodb(self, datasets):
        logger.info(f'Loading hidden network layout from mongodb for datasets {datasets}')
        for dataset in datasets:
            try:
                layout = self._load_layout_from_mongodb(dataset, 'hidden_network_layout')
                self._hidden_network_layouts[dataset] = layout
            except Exception as ex:
                logger.error(f'Error loading hidden network layout for dataset {dataset} with error {ex}')

    def _load_layout_from_mongodb(self, dataset, collection_name):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection(collection_name)
        layout = collection.aggregate_pandas_all([{'$project': {'_id': 0}}])
        client.close()
        return layout

    def prepopulate(self):
        logger.info('Prepopulating propagation factory')
        self.persist(self.available_datasets)
        logger.info('Prepopulating egonet')
        self.egonet.persist(self.available_datasets)
        logger.info('Prepopulating network metrics')
        self.network_metrics.persist(self.available_datasets)
        logger.info('Prepopulating diffusion metrics')
        self.diffusion_metrics.persist(self.available_datasets)
        logger.info('Done prepopulating propagation factory')


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
