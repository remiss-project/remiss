import time
from abc import ABC
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
                 simplification=None, threshold=0.2, delete_vertices=True, k_cores=4, available_datasets=None):
        super().__init__(host, port, database, available_datasets)
        self.delete_vertices = delete_vertices
        self.threshold = threshold
        self.reference_types = reference_types
        self._hidden_networks = {}
        self._simplified_hidden_networks = {}
        self.layout = layout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.simplification = simplification
        self.k_cores = k_cores

    def get_egonet(self, dataset, user, depth):
        hidden_network = self.get_hidden_network(dataset)
        if user:
            try:
                node = hidden_network.vs.find(username=user)
                egonet = hidden_network.induced_subgraph(hidden_network.neighborhood(node, order=depth))
                return egonet
            except (RuntimeError, ValueError) as ex:
                print(f'Computing neighbourhood for user {user} failed, computing the whole network')
        if self.simplification:
            print(f'Returning {self.simplification} simplified hidden network')
        return self._simplified_hidden_networks[dataset] if self.simplification else hidden_network

    def load_from_cache(self, dataset):
        cache_dir = self.cache_dir / dataset
        graph_file = cache_dir / 'hidden_network_graph.graphmlz'
        layout_file = cache_dir / 'hidden_network_layout.feather'
        if self.simplification:
            simplified_graph_file = cache_dir / f'hidden_network_graph-{self.simplification}-{self.threshold}.graphmlz'
            simplified_layout_file = cache_dir / f'hidden_network_layout-{self.simplification}-{self.threshold}.feather'
            simplified_graph = ig.Graph.Read_GraphMLz(str(simplified_graph_file))
            simplified_layout = pd.read_feather(str(simplified_layout_file))
            simplified_graph['layout_df'] = simplified_layout
            self._simplified_hidden_networks[dataset] = simplified_graph
            print(f'Loaded simplified hidden network from {simplified_graph_file}')
            print(self._simplified_hidden_networks[dataset].summary())
        graph = ig.Graph.Read_GraphMLz(str(graph_file))
        layout = pd.read_feather(str(layout_file))
        graph['layout_df'] = layout
        self._hidden_networks[dataset] = graph
        print(f'Loaded hidden network from {graph_file}')
        print(self._hidden_networks[dataset].summary())

    def compute_and_store(self, dataset):
        (self.cache_dir / dataset).mkdir(parents=True, exist_ok=True)
        graph_file = self.cache_dir / dataset / 'hidden_network_graph.graphmlz'
        layout_file = self.cache_dir / dataset / 'hidden_network_layout.feather'
        graph = self._compute_hidden_network(dataset)
        graph.write_graphmlz(str(graph_file))
        layout = self.compute_layout(graph)
        layout.to_feather(str(layout_file))
        graph['layout_df'] = layout
        self._hidden_networks[dataset] = graph
        if self.simplification:
            simplified_graph_file = self.cache_dir / dataset / f'hidden_network_graph-{self.simplification}-{self.threshold}.graphmlz'
            simplified_layout_file = self.cache_dir / dataset / f'hidden_network_layout-{self.simplification}-{self.threshold}.feather'
            simplified_graph = self._simplify_graph(self._hidden_networks[dataset])
            simplified_layout = self.compute_layout(simplified_graph)
            simplified_graph.write_graphmlz(str(simplified_graph_file))
            simplified_layout.to_feather(str(simplified_layout_file))
            simplified_graph['layout_df'] = simplified_layout
            self._simplified_hidden_networks[dataset] = simplified_graph
            print(f'Simplified hidden network saved to {simplified_graph_file}')

    def is_cached(self, dataset):
        dataset_dir = self.cache_dir / dataset
        hn_graph_file = dataset_dir / 'hidden_network_graph.graphmlz'
        hn_layout_file = dataset_dir / 'hidden_network_layout.feather'
        is_cached = hn_graph_file.exists() and hn_layout_file.exists()
        if self.simplification:
            simplified_hn_graph_file = dataset_dir / f'hidden_network_graph-{self.simplification}-{self.threshold}.graphmlz'
            simplified_hn_layout_file = dataset_dir / f'hidden_network_layout-{self.simplification}-{self.threshold}.feather'
            is_cached = is_cached and simplified_hn_graph_file.exists() and simplified_hn_layout_file
        return is_cached

    def get_hidden_network(self, dataset):
        if dataset not in self._hidden_networks:
            if self.cache_dir:
                if self.is_cached(dataset):
                    self.load_from_cache(dataset)
                else:
                    self.compute_and_store(dataset)
            else:
                self._hidden_networks[dataset] = self._compute_hidden_network(dataset)
                if self.simplification:
                    self._simplified_hidden_networks[dataset] = self._simplify_graph(self._hidden_networks[dataset])
        return self._hidden_networks[dataset]

    def _get_authors(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party'}},
            {'$unionWith': {'coll': dataset, 'pipeline': [
                {'$project': {'id': '$author.id',
                              'username': '$author.username',
                              'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                              'party': '$author.remiss_metadata.party'}}]}},
            {'$group': {'_id': '$id',
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'}}},
            {'$project': {'_id': 0,
                          'id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]}}}
        ]
        print('Computing authors')
        start_time = time.time()
        authors = collection.aggregate_pandas_all(node_pipeline)
        print(f'Authors computed in {time.time() - start_time} seconds')
        return authors

    def get_legitimacy(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': '$author.id',
                        'username': {'$first': '$author.username'},
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'id': '$_id',
                          'username': 1,
                          'legitimacy': 1}},
        ]
        print('Computing legitimacy')
        start_time = time.time()
        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        legitimacy = legitimacy.set_index('id')
        legitimacy = legitimacy.sort_values('legitimacy', ascending=False)
        print(f'Legitimacy computed in {time.time() - start_time} seconds')
        return legitimacy

    def get_reputation(self, dataset, unit='day', bin_size=1):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': {'author': '$author.id',
                                'date': {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}}},
                        'username': {'$first': '$author.username'},
                        'reputation': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'id': '$_id.author',
                          'date': '$_id.date',
                          'username': 1,
                          'reputation': 1}},
        ]
        print('Computing reputation')
        start_time = time.time()
        reputation = collection.aggregate_pandas_all(node_pipeline)
        reputation = reputation.pivot(columns='date', index=['id', 'username'], values='reputation')
        reputation = reputation.cumsum(axis=1)
        print(f'Reputation computed in {time.time() - start_time} seconds')
        return reputation

    def _get_references(self, dataset):
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

        print('Computing graph')
        start_time = time.time()
        # switch id by position (which will be the node id in the graph) and set it as index
        author_to_id = authors['id'].reset_index().set_index('id')
        # convert references which are author id based to graph id based
        references['source'] = author_to_id.loc[references['source']].reset_index(drop=True)
        references['target'] = author_to_id.loc[references['target']].reset_index(drop=True)

        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['id_'] = authors['id']
        g.vs['username'] = authors['username']
        g.vs['is_usual_suspect'] = authors['is_usual_suspect']
        g.vs['party'] = authors['party']
        g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
        g.es['weight'] = references['weight']
        g.es['weight_inv'] = references['weight_inv']
        g.es['weight_norm'] = references['weight_norm']
        print(g.summary())
        print(f'Graph computed in {time.time() - start_time} seconds')

        return g

    def plot_egonet(self, collection, user, depth):
        network = self.get_egonet(collection, user, depth)
        network = network.as_undirected(mode='collapse')

        return self.plot_network(network)

    def plot_hidden_network(self, collection):
        network = self.get_hidden_network(collection)
        return self.plot_network(network)

    def compute_layout(self, network):
        print(f'Computing {self.layout} layout')
        start_time = time.time()
        layout = network.layout(self.layout, dim=3)
        print(f'Layout computed in {time.time() - start_time} seconds')
        layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
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

    def plot_network(self, network):
        if 'layout_df' not in network.attributes():
            layout = self.compute_layout(network)
        else:
            layout = network['layout_df']
        print('Computing plot for network')
        print(network.summary())
        start_time = time.time()
        edges = pd.DataFrame(network.get_edgelist(), columns=['source', 'target'])
        edge_positions = layout.iloc[edges.values.flatten()].reset_index(drop=True)
        nones = edge_positions[1::2].assign(x=None, y=None, z=None)
        edge_positions = pd.concat([edge_positions, nones]).sort_index().reset_index(drop=True)

        metadata = pd.DataFrame({'is_usual_suspect': network.vs['is_usual_suspect'], 'party': network.vs['party']})

        color_map = {(False, False): 'blue',
                     (False, True): 'yellow',
                     (True, False): 'red',
                     (True, True): 'purple'}

        color = metadata.apply(lambda x: color_map[(x['is_usual_suspect'], x['party'] is not None)], axis=1)

        edge_trace = go.Scatter3d(x=edge_positions['x'],
                                  y=edge_positions['y'],
                                  z=edge_positions['z'],
                                  mode='lines',
                                  line=dict(color='rgb(125,125,125)', width=1),
                                  hoverinfo='none'
                                  )

        text = []
        for node in network.vs:
            is_usual_suspect = 'Yes' if node['is_usual_suspect'] else 'No'
            party = f'Party: {node["party"]}' if node['party'] else '-'
            text.append(f'{node["username"]}<br>'
                        f'Usual suspect: {is_usual_suspect}<br>'
                        f'Party: {party}')

        node_trace = go.Scatter3d(x=layout['x'],
                                  y=layout['y'],
                                  z=layout['z'],
                                  mode='markers',
                                  name='users',
                                  marker=dict(symbol='circle',
                                              size=6 if len(network.vs) < 100 else 2,
                                              color=color,
                                              colorscale='Viridis',
                                              line=dict(color='rgb(50,50,50)', width=0.5)
                                              ),
                                  text=text,
                                  hovertemplate='%{text}',
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
            eye=dict(x=0.7, y=0.7, z=0.7)
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
