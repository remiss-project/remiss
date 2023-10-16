from abc import ABC
from datetime import datetime

import networkx
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import pymongoarrow.monkey
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongoarrow.schema import Schema
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

pymongoarrow.monkey.patch_all()


class MongoPlotFactory(ABC):
    def __init__(self, host="localhost", port=27017, database="test_remiss"):
        super().__init__()
        self.host = host
        self.port = port
        self.database = database
        self._available_datasets = None
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

    @staticmethod
    def _get_hashtag_freqs(collection):
        available_hashtags_freqs = list(collection.aggregate([
            {'$unwind': '$entities.hashtags'},
            {'$group': {'_id': '$entities.hashtags.tag', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]))
        available_hashtags_freqs = [(x['_id'], x['count']) for x in available_hashtags_freqs]
        return available_hashtags_freqs


class TweetUserPlotFactory(MongoPlotFactory):

    def plot_tweet_series(self, collection, hashtag, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]
        return self._get_count_area_plot(pipeline, collection, hashtag, start_time, end_time)

    def plot_user_series(self, collection, hashtag, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "users": {'$addToSet': "$author.username"}}
            },
            {'$project': {'count': {'$size': '$users'}}},
            {'$sort': {'_id': 1}}
        ]
        return self._get_count_area_plot(pipeline, collection, hashtag, start_time, end_time)

    def _get_count_area_plot(self, pipeline, collection, hashtag, start_time, end_time):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        self._validate_collection(database, collection)
        collection = database.get_collection(collection)

        pipeline = self._add_filters(pipeline, hashtag, start_time, end_time)

        df = collection.aggregate_pandas_all(
            pipeline,
            schema=Schema({'_id': datetime, 'count': int})
        )
        df = df.rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time').squeeze()

        return px.area(df, labels={"value": "Count"})

    @staticmethod
    def _add_filters(pipeline, hashtag, start_time, end_time):
        if hashtag:
            pipeline.insert(0, {'$match': {'entities.hashtags.tag': hashtag}})
        if start_time:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            pipeline.insert(0, {'$match': {'created_at': {'$lte': end_time}}})
        return pipeline


class EgonetPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, database="test_remiss",
                 reference_types=('replied_to', 'quoted', 'retweeted'), layout='fruchterman_reingold'):
        super().__init__(host, port, database)
        self.reference_types = reference_types
        self._hidden_networks = {}
        self.layout = layout

    def get_egonet(self, dataset, user, depth):
        egonet = self.compute_hidden_network(dataset)
        if user:
            try:
                user_id = self.get_user_id(dataset, user)
                egonet = nx.ego_graph(egonet, user_id, depth, center=True, undirected=True)
            except RuntimeError as ex:
                print(f'Computing neighbourhood for user {user} failed, computing the whole network')

        return egonet

    def compute_hidden_network(self, dataset):
        if dataset not in self._hidden_networks:
            self._hidden_networks[dataset] = self._compute_hidden_network(dataset)
        return self._hidden_networks[dataset]

    def _compute_hidden_network(self, dataset):
        """
        Computes the hidden graph, this is, the graph of users that have interacted with each other.
        :param dataset: collection name within the database where the tweets are stored
        :return: a networkx graph with the users as nodes and the edges representing interactions between users
        """
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        collection = database.get_collection(dataset)

        graph = networkx.DiGraph()
        with logging_redirect_tqdm():
            for tweet in tqdm(collection.find()):
                if 'referenced_tweets' in tweet:
                    source = tweet['author']
                    for referenced_tweet in tweet['referenced_tweets']:
                        if referenced_tweet['type'] in self.reference_types:
                            target = self.get_reference_target_user(referenced_tweet, collection)
                            if source['id'] != target['id']:
                                if not graph.has_node(source['id']):
                                    graph.add_node(source['id'], **source)
                                if not graph.has_node(target['id']):
                                    graph.add_node(target['id'], **target)
                                graph.add_edge(source['id'], target['id'])

                        else:
                            print(f'Tweet {tweet["id"]} has an unknown reference type {referenced_tweet["type"]}')

        client.close()

        return graph

    @staticmethod
    def get_reference_target_user(referenced_tweet, collection):
        if 'author' in referenced_tweet:
            target = referenced_tweet['author']
        else:
            referenced_tweet_id = referenced_tweet['id']
            referenced_tweet = collection.find_one({'id': referenced_tweet_id})
            if referenced_tweet:
                target = referenced_tweet['author']
            else:
                print(f'Referenced tweet {referenced_tweet_id} not found')
                target = {'id': referenced_tweet_id}
        return target

    def plot_egonet(self, collection, user, depth):
        network = self.get_egonet(collection, user, depth)
        return self.plot_network(network)

    def plot_network(self, network):
        if self.layout == 'fruchterman_reingold':
            layout = nx.fruchterman_reingold_layout(network)
        else:
            raise ValueError(f'Unknown layout {self.layout}')

        edge_x = []
        edge_y = []
        for edge in network.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in network.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)

        node_colors = []
        node_text = []
        # Color code
        # 0 - green: not usual suspect nor politician
        # 1 - red: usual suspect
        # 2- yellow: politician
        # 3 - purple: usual suspect and politician
        for node in network:
            try:
                username = network.nodes[node]['username']
                label = f'{username}'
                color = 'green'
                if 'remiss_metadata' in network.nodes[node]:
                    is_usual_suspect = network.nodes[node]['remiss_metadata']['is_usual_suspect']
                    party = network.nodes[node]['remiss_metadata']['party']
                    if is_usual_suspect and party:
                        label = f'{username}: usual suspect from {party}'
                        color = 'purple'
                    elif is_usual_suspect:
                        label = f'{username}: usual suspect'
                        color = 'red'
                    elif party:
                        label = f'{username}: {party}'
                        color = 'yellow'

            except KeyError:
                label = node
                color = 'green'
            node_text.append(label)
            node_colors.append(color)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                size=10,
                line_width=2,
                color=node_colors,
            ),
            text=node_text,
            showlegend=True
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )

        return fig
