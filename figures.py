import time
from abc import ABC
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dash_table import DataTable
import pymongoarrow.monkey
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongoarrow.schema import Schema
from tqdm import tqdm

pymongoarrow.monkey.patch_all()


class MongoPlotFactory(ABC):
    def __init__(self, host="localhost", port=27017, database="test_remiss", available_datasets=None):
        super().__init__()
        self.host = host
        self.port = port
        self.database = database
        self._available_datasets = available_datasets
        self._min_max_dates = {}
        self._available_hashtags = {}

    def _validate_collection(self, database, collection):
        try:
            database.validate_collection(collection)
        except OperationFailure as ex:
            if ex.details['code'] == 26:
                raise ValueError(f'Dataset {collection} does not exist') from ex
            else:
                raise ex

    def get_date_range(self, collection):
        if collection not in self._min_max_dates:
            client = MongoClient(self.host, self.port)
            database = client.get_database(self.database)
            self._validate_collection(database, collection)
            collection = database.get_collection(collection)
            self._min_max_dates[collection] = self._get_date_range(collection)
            client.close()
        return self._min_max_dates[collection]

    def get_hashtag_freqs(self, collection):
        if collection not in self._available_hashtags:
            client = MongoClient(self.host, self.port)
            database = client.get_database(self.database)
            self._validate_collection(database, collection)
            collection = database.get_collection(collection)
            self._available_hashtags[collection] = self._get_hashtag_freqs(collection)
            client.close()
        return self._available_hashtags[collection]

    def get_users(self, collection):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        dataset = database.get_collection(collection)
        available_users = [str(x) for x in dataset.distinct('author.username')]
        client.close()
        return available_users

    def get_user_id(self, dataset, username):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)
        tweet = collection.find_one({'author.username': username})
        client.close()
        if tweet:
            return tweet['author']['id']
        else:
            raise RuntimeError(f'User {username} not found')

    @property
    def available_datasets(self):
        if self._available_datasets is None:
            client = MongoClient(self.host, self.port)
            self._available_datasets = client.get_database(self.database).list_collection_names()
        return self._available_datasets

    @staticmethod
    def _get_date_range(collection):
        min_date_allowed = collection.find_one(sort=[('created_at', 1)])['created_at'].date()
        max_date_allowed = collection.find_one(sort=[('created_at', -1)])['created_at'].date()
        return min_date_allowed, max_date_allowed

    def _get_hashtag_freqs(self, collection):
        pipeline = [
            {'$unwind': '$entities.hashtags'},
            {'$group': {'_id': '$entities.hashtags.tag', 'count': {'$count': {}}}},
            {'$sort': {'count': -1}}
        ]
        available_hashtags_freqs = list(collection.aggregate(pipeline))
        available_hashtags_freqs = [(x['_id'], x['count']) for x in available_hashtags_freqs]
        return available_hashtags_freqs


class TweetUserPlotFactory(MongoPlotFactory):

    def plot_tweet_series(self, collection, hashtags, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]
        print('Computing tweet series')
        start_computing_time = time.time()
        plot = self._get_count_area_plot(pipeline, collection, hashtags, start_time, end_time)
        print(f'Tweet series computed in {time.time() - start_computing_time} seconds')
        return plot

    def plot_user_series(self, collection, hashtags, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "users": {'$addToSet': "$author.username"}}
            },
            {'$project': {'count': {'$size': '$users'}}},
            {'$sort': {'_id': 1}}
        ]
        print('Computing user series')
        start_computing_time = time.time()
        plot = self._get_count_area_plot(pipeline, collection, hashtags, start_time, end_time)
        print(f'User series computed in {time.time() - start_computing_time} seconds')
        return plot

    def _perform_count_aggregation(self, pipeline, collection):
        df = collection.aggregate_pandas_all(
            pipeline,
            schema=Schema({'_id': datetime, 'count': int})
        )
        df = df.rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')

        return df

    def _get_count_data(self, pipeline, hashtags, start_time, end_time, collection):
        normal_pipeline = self._add_filters(pipeline, hashtags, start_time, end_time, user_type='normal')
        normal_df = self._perform_count_aggregation(normal_pipeline, collection)

        suspect_pipeline = self._add_filters(pipeline, hashtags, start_time, end_time, user_type='suspect')
        suspect_df = self._perform_count_aggregation(suspect_pipeline, collection)

        politician_pipeline = self._add_filters(pipeline, hashtags, start_time, end_time, user_type='politician')
        politician_df = self._perform_count_aggregation(politician_pipeline, collection)

        suspect_politician_pipeline = self._add_filters(pipeline, hashtags, start_time, end_time,
                                                        user_type='suspect_politician')
        suspect_politician_df = self._perform_count_aggregation(suspect_politician_pipeline, collection)

        df = pd.concat([normal_df, suspect_df, politician_df, suspect_politician_df], axis=1)
        df.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']

        return df

    def _get_count_area_plot(self, pipeline, collection, hashtags, start_time, end_time):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        self._validate_collection(database, collection)
        collection = database.get_collection(collection)

        df = self._get_count_data(pipeline, hashtags, start_time, end_time, collection)

        if len(df) == 1:
            plot = px.bar(df, labels={"value": "Count"})
        else:
            plot = px.area(df, labels={"value": "Count"})

        return plot

    @staticmethod
    def _add_filters(pipeline, hashtags, start_time, end_time, user_type):
        pipeline = pipeline.copy()
        if user_type == 'normal':
            pipeline.insert(0, {'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                           'author.remiss_metadata.party': None}})
        elif user_type == 'suspect':
            pipeline.insert(0, {'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                           'author.remiss_metadata.party': None}})
        elif user_type == 'politician':
            pipeline.insert(0, {'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                           'author.remiss_metadata.party': {'$ne': None}}})
        elif user_type == 'suspect_politician':
            pipeline.insert(0, {'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                           'author.remiss_metadata.party': {'$ne': None}}})
        else:
            raise ValueError(f'Unknown user type {user_type}')

        if hashtags:
            for hashtag in hashtags:
                pipeline.insert(0, {'$match': {'entities.hashtags.tag': hashtag}})
        if start_time:
            start_time = pd.to_datetime(start_time)
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            end_time = pd.to_datetime(end_time)
            pipeline.insert(0, {'$match': {'created_at': {'$lte': end_time}}})
        return pipeline


class EgonetPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, database="test_remiss", cache_dir=None,
                 reference_types=('replied_to', 'quoted', 'retweeted'), layout='fruchterman_reingold',
                 simplification=None, threshold=0.2, delete_vertices=True, k_cores=4, frequency='1D',
                 available_datasets=None, prepopulate=False):
        super().__init__(host, port, database, available_datasets)
        self.frequency = frequency
        self.bin_size = int(frequency[:-1])
        pd_units = {'D': 'day', 'W': 'week', 'M': 'month', 'Y': 'year'}
        self.unit = pd_units[frequency[-1]]
        self.delete_vertices = delete_vertices
        self.threshold = threshold
        self.reference_types = reference_types
        self._hidden_networks = {}
        self._simplified_hidden_networks = {}
        self.layout = layout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.simplification = simplification
        self.k_cores = k_cores
        self.prepopulate = prepopulate
        if self.prepopulate:
            self.prepopulate_cache()

    def plot_egonet(self, collection, user, depth, start_date, end_date):
        network = self.get_egonet(collection, user, depth)
        network = network.as_undirected(mode='collapse')

        return self.plot_network(network, start_date, end_date)

    def get_egonet(self, dataset, user, depth):
        """
        Returns the egonet of a user of a certain date and depth if present,
        otherwise returns the simplified hidden network
        :param dataset:
        :param user:
        :param depth:
        :return:
        """
        hidden_network = self.get_hidden_network(dataset)
        # check if the user is in the hidden network
        if user:
            try:
                node = hidden_network.vs.find(username=user)
                neighbours = hidden_network.neighborhood(node, order=depth)
                egonet = hidden_network.induced_subgraph(neighbours)
                return egonet
            except (RuntimeError, ValueError) as ex:
                print(f'Computing neighbourhood for user {user} failed with error {ex}')
        return self.get_simplified_hidden_network(dataset)

    def get_hidden_network(self, dataset):
        if dataset not in self._hidden_networks:
            if self.cache_dir and self.is_cached(dataset, 'hidden_network'):
                network = self.load_from_cache(dataset, 'hidden_network')
            else:
                network = self._compute_hidden_network(dataset)
                if self.cache_dir:
                    self.save_to_cache(dataset, network, 'hidden_network')
            self._hidden_networks[dataset] = network

        return self._hidden_networks[dataset]

    def get_simplified_hidden_network(self, dataset):
        if dataset not in self._simplified_hidden_networks:
            if self.cache_dir and self.is_cached(dataset, 'simplified_hidden_network'):
                network = self.load_from_cache(dataset, 'simplified_hidden_network')
            else:
                network = self.get_hidden_network(dataset)
                network = self._simplify_graph(network)
                if self.cache_dir:
                    self.save_to_cache(dataset, network, 'simplified_hidden_network')
            self._simplified_hidden_networks[dataset] = network

        return self._simplified_hidden_networks[dataset]

    def is_cached(self, dataset, stem):
        dataset_dir = self.cache_dir / dataset
        hn_graph_file = dataset_dir / f'{stem}.graphml'
        hn_layout_file = dataset_dir / f'{stem}.layout.csv'
        return hn_graph_file.exists() and hn_layout_file.exists()

    def load_from_cache(self, dataset, stem):
        dataset_dir = self.cache_dir / dataset
        hn_graph_file = dataset_dir / f'{stem}.graphml'
        hn_layout_file = dataset_dir / f'{stem}.layout.csv'
        network = ig.read(hn_graph_file)
        layout = pd.read_csv(hn_layout_file)
        network['layout_df'] = layout
        return network

    def save_to_cache(self, dataset, network, stem):
        dataset_dir = self.cache_dir / dataset
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        hn_graph_file = dataset_dir / f'{stem}.graphml'
        hn_layout_file = dataset_dir / f'{stem}.layout.csv'
        network.write_graphml(str(hn_graph_file))
        layout = network['layout_df']
        layout.to_csv(hn_layout_file, index=False)

    def prepopulate_cache(self):
        if not self.cache_dir:
            raise ValueError('Cache directory not set')

        for dataset in (pbar := tqdm(self.available_datasets, desc='Prepopulating cache')):
            pbar.set_postfix_str(dataset)
            self.get_hidden_network(dataset)

    def get_legitimacy(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': '$author.id',
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'legitimacy': 1}},
        ]
        print('Computing legitimacy')
        start_time = time.time()
        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        legitimacy = legitimacy.set_index('author_id')
        legitimacy = legitimacy.sort_values('legitimacy', ascending=False)
        print(f'Legitimacy computed in {time.time() - start_time} seconds')
        return legitimacy

    def _get_legitimacy_per_time(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': {'author': '$author.id',
                                'date': {
                                    "$dateTrunc": {'date': "$created_at", 'unit': self.unit, 'binSize': self.bin_size}}
                                },
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id.author',
                          'date': '$_id.date',
                          'legitimacy': 1}},
        ]
        print('Computing reputation')

        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        if len(legitimacy) == 0:
            raise ValueError(f'No data available for the selected time range and dataset: {dataset} {self.unit} {self.bin_size}')
        legitimacy = legitimacy.pivot(columns='date', index='author_id', values='legitimacy')
        legitimacy = legitimacy.fillna(0)
        return legitimacy

    def get_reputation(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_per_time(dataset)
        reputation = legitimacy.cumsum(axis=1)

        print(f'Reputation computed in {time.time() - start_time} seconds')
        return reputation

    def get_status(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_per_time(dataset)
        reputation = legitimacy.cumsum(axis=1)
        status = reputation.apply(lambda x: x.argsort())
        print(f'Status computed in {time.time() - start_time} seconds')
        return status

    def _add_date_filters(self, pipeline, start_date, end_date):
        if start_date:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})
        if end_date:
            pipeline.insert(0, {'$match': {'created_at': {'$lt': pd.to_datetime(end_date)}}})

    def _get_authors(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)
        nested_pipeline = [
            {'$project': {'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party'}}]
        self._add_date_filters(nested_pipeline, start_date, end_date)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'author_id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party'}},
            {'$unionWith': {'coll': dataset, 'pipeline': nested_pipeline}},  # Fetch missing authors
            {'$group': {'_id': '$author_id',
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]}}}
        ]
        self._add_date_filters(node_pipeline, start_date, end_date)
        print('Computing authors')
        start_time = time.time()
        schema = Schema({'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str})
        authors = collection.aggregate_pandas_all(node_pipeline, schema=schema)
        print(f'Authors computed in {time.time() - start_time} seconds')
        return authors

    def _get_references(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        references_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'source': '$author.id', 'target': '$referenced_tweets.author.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'},
                        'weight': {'$count': {}}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target', 'weight': 1}},
            {'$group': {'_id': '$source',
                        'node_weight': {'$sum': '$weight'},
                        'references': {'$push': {'target': '$target', 'weight': '$weight'}}}},
            {'$unwind': '$references'},
            {'$project': {'_id': 0, 'source': '$_id', 'target': '$references.target',
                          'weight': '$references.weight',
                          'weight_inv': {'$divide': [1, '$references.weight']},
                          'weight_norm': {'$divide': ['$references.weight', '$node_weight']},
                          }},
        ]
        self._add_date_filters(references_pipeline, start_date, end_date)
        print('Computing references')
        start_time = time.time()
        references = collection.aggregate_pandas_all(references_pipeline)
        print(f'References computed in {time.time() - start_time} seconds')
        client.close()
        return references

    def _compute_hidden_network(self, dataset):
        """
        Computes the hidden graph, this is, the graph of users that have interacted with each other.
        :param dataset: collection name within the database where the tweets are stored
        :return: a networkx graph with the users as nodes and the edges representing interactions between users
        """
        authors = self._get_authors(dataset)
        references = self._get_references(dataset)
        available_reputation = self.get_reputation(dataset)
        available_legitimacy = self.get_legitimacy(dataset)
        if len(authors) == 0:
            # in case of no authors we return an empty graph
            return ig.Graph(directed=True)

        print('Computing graph')
        start_time = time.time()
        # switch id by position (which will be the node id in the graph) and set it as index
        author_to_id = authors['author_id'].reset_index().set_index('author_id')
        # convert references which are author id based to graph id based
        references['source'] = author_to_id.loc[references['source']].reset_index(drop=True)
        references['target'] = author_to_id.loc[references['target']].reset_index(drop=True)
        # we only have reputation and legitimacy for a subset of the authors, so the others will be set to nan
        reputation = pd.DataFrame(np.nan, index=author_to_id.index, columns=available_reputation.columns)
        reputation.loc[available_reputation.index] = available_reputation
        legitimacy = pd.Series(np.nan, index=author_to_id.index)
        legitimacy[available_legitimacy.index] = available_legitimacy['legitimacy']

        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['author_id'] = authors['author_id']
        g.vs['username'] = authors['username']
        g.vs['is_usual_suspect'] = authors['is_usual_suspect']
        g.vs['party'] = authors['party']
        g['reputation'] = reputation
        g.vs['legitimacy'] = legitimacy
        g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
        g.es['weight'] = references['weight']
        g.es['weight_inv'] = references['weight_inv']
        g.es['weight_norm'] = references['weight_norm']
        print(g.summary())
        print(f'Graph computed in {time.time() - start_time} seconds')

        layout = self.compute_layout(g)
        g['layout'] = layout

        return g

    def compute_layout(self, network):
        print(f'Computing {self.layout} layout')
        start_time = time.time()
        layout = network.layout(self.layout, dim=3)
        print(f'Layout computed in {time.time() - start_time} seconds')
        return layout

    def _simplify_graph(self, network):
        if self.simplification == 'maximum_spanning_tree':
            network = network.spanning_tree(weights=network.es['weight_inv'])
        elif self.simplification == 'k_core':
            network = network.k_core(self.k_cores)
        elif self.simplification == 'backbone':
            network = compute_backbone(network, self.threshold, self.delete_vertices)
        else:
            raise ValueError(f'Unknown simplification {self.simplification}')
        return network

    def plot_network(self, network, start_date=None, end_date=None):
        if 'layout' not in network.attributes():
            layout = self.compute_layout(network)
        else:
            layout = network['layout']
        layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
        print('Computing plot for network')
        print(network.summary())
        start_time = time.time()
        edges = pd.DataFrame(network.get_edgelist(), columns=['source', 'target'])
        edge_positions = layout.iloc[edges.values.flatten()].reset_index(drop=True)
        nones = edge_positions[1::2].assign(x=None, y=None, z=None)
        edge_positions = pd.concat([edge_positions, nones]).sort_index().reset_index(drop=True)

        # Legitimacy -> vertex color
        # Reputation -> vertex size
        # Party / Usual suspect -> vertex marker

        metadata = pd.DataFrame({'is_usual_suspect': network.vs['is_usual_suspect'], 'party': network.vs['party']})

        marker_map = {(False, False): 'circle',
                      (False, True): 'square',
                      (True, False): 'diamond',
                      (True, True): 'cross'}

        markers = metadata.apply(lambda x: marker_map[(x['is_usual_suspect'], x['party'] is not None)], axis=1)

        if start_date:
            size = network['reputation'][start_date]
        else:
            size = network['reputation'].mean(axis=1)
        # Add 1 offset and set 1 as minimum size
        size += 4
        size = size.fillna(1)
        size = size / size.max() * 50 if len(network.vs) > 100 else size / size.max() * 100

        color = pd.Series(network.vs['legitimacy'])
        # color = color.fillna(0)

        edge_trace = go.Scatter3d(x=edge_positions['x'],
                                  y=edge_positions['y'],
                                  z=edge_positions['z'],
                                  mode='lines',
                                  line=dict(color='rgb(125,125,125)', width=1),
                                  hoverinfo='none',
                                  name='Interactions',
                                  showlegend=False
                                  )

        text = []
        for node in network.vs:
            is_usual_suspect = 'Yes' if node['is_usual_suspect'] else 'No'
            party = f'Party: {node["party"]}' if node['party'] else '-'
            legitimacy_value = node["legitimacy"] if not np.isnan(node["legitimacy"]) else '-'
            reputation_value = network["reputation"].loc[node['author_id']]
            reputation_value = reputation_value[start_date] if start_date else reputation_value.mean()
            reputation_value = f'{reputation_value:.2f}' if not np.isnan(reputation_value) else '-'

            node_text = f'Username: {node["username"]}<br>' \
                        f'Is usual suspect: {is_usual_suspect}<br>' \
                        f'Party: {party}<br>' \
                        f'Legitimacy: {legitimacy_value}<br>' \
                        f'Reputation: {reputation_value}'
            text.append(node_text)

        node_trace = go.Scatter3d(x=layout['x'],
                                  y=layout['y'],
                                  z=layout['z'],
                                  mode='markers',
                                  marker=dict(symbol=markers,
                                              size=size,
                                              color=color,
                                              # coloscale set to $champagne: #ffead0ff;
                                              # to $bright-pink-crayola: #f76f8eff;
                                              colorscale=[[0, 'rgb(255, 234, 208)'], [1, 'rgb(247, 111, 142)']],
                                              colorbar=dict(thickness=20, title='Legitimacy'),
                                              line=dict(color='rgb(50,50,50)', width=0.5),
                                              ),
                                  text=text,
                                  hovertemplate='%{text}',
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
            showlegend=False,
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

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        fig.update_layout(scene_camera=camera)
        print(f'Plot computed in {time.time() - start_time} seconds')
        return fig


def compute_backbone(graph, alpha=0.05, delete_vertices=True):
    # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
    weights = np.array(graph.es['weight_norm'])
    degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
    alphas = (1 - weights) ** (degrees - 1)
    good = np.nonzero(alphas > alpha)[0]
    backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)
    return backbone


class TopTableFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, database="test_remiss", available_datasets=None, limit=50,
                 retweet_table_columns=None, user_table_columns=None):
        super().__init__(host, port, database, available_datasets)
        self.limit = limit
        self.top_table_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party']
        self.retweeted_table_columns = ['id', 'text',
                                        'count'] if retweet_table_columns is None else retweet_table_columns
        self.user_table_columns = ['username', 'count'] if user_table_columns is None else user_table_columns

    def get_top_retweeted(self, collection, start_time=None, end_time=None):
        pipeline = [
            {'$group': {'_id': '$id', 'text': {'$first': '$text'}, 'count': {'$count': {}}}},
            {'$sort': {'count': -1}},
            {'$limit': self.limit},
            {'$project': {'_id': 0, 'id': '$_id', 'text': 1, 'count': 1}}
        ]
        pipeline = self._add_filters(pipeline, start_time, end_time)
        df = self._perform_top_aggregation(pipeline, collection)[self.retweeted_table_columns]
        return df

    def get_top_users(self, collection, start_time=None, end_time=None):
        pipeline = [
            {'$group': {'_id': '$author.username', 'count': {'$count': {}}}},
            {'$sort': {'count': -1}},
            {'$limit': self.limit},
            {'$project': {'_id': 0, 'username': '$_id', 'count': 1}}
        ]
        pipeline = self._add_filters(pipeline, start_time, end_time)
        df = self._perform_top_aggregation(pipeline, collection)[self.user_table_columns]
        return df

    def get_top_table_data(self, collection, start_time=None, end_time=None):
        pipeline = [
            {'$group': {'_id': '$text', 'User': {'$first': '$author.username'},
                        'tweet_id': {'$first': '$id'},
                        'Retweets': {'$max': '$public_metrics.retweet_count'},
                        'Is usual suspect': {'$max': '$author.remiss_metadata.is_usual_suspect'},
                        'Party': {'$max': '$author.remiss_metadata.party'}}},
            {'$sort': {'Retweets': -1}},
            {'$limit': self.limit},
            {'$project': {'_id': 0, 'tweet_id': 1, 'User': 1, 'Text': '$_id', 'Retweets': 1, 'Is usual suspect': 1,
                          'Party': 1}}

        ]
        pipeline = self._add_filters(pipeline, start_time, end_time)
        df = self._perform_top_aggregation(pipeline, collection)
        df = df.set_index('tweet_id')
        return df

    def _add_filters(self, pipeline, start_time=None, end_time=None):
        pipeline = pipeline.copy()
        if start_time:
            start_time = pd.to_datetime(start_time)
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            end_time = pd.to_datetime(end_time)
            pipeline.insert(0, {'$match': {'created_at': {'$lte': end_time}}})
        return pipeline

    def _perform_top_aggregation(self, pipeline, collection):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        dataset = database.get_collection(collection)
        top_prolific = dataset.aggregate_pandas_all(pipeline)
        client.close()
        return top_prolific
