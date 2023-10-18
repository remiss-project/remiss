from abc import ABC
from datetime import datetime

import igraph as ig
import pandas as pd
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

    def _perform_count_aggregation(self, pipeline, collection):
        df = collection.aggregate_pandas_all(
            pipeline,
            schema=Schema({'_id': datetime, 'count': int})
        )
        df = df.rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')

        return df

    def _get_count_data(self, pipeline, hashtag, start_time, end_time, collection):
        normal_pipeline = self._add_filters(pipeline, hashtag, start_time, end_time, user_type='normal')
        normal_df = self._perform_count_aggregation(normal_pipeline, collection)

        suspect_pipeline = self._add_filters(pipeline, hashtag, start_time, end_time, user_type='suspect')
        suspect_df = self._perform_count_aggregation(suspect_pipeline, collection)

        politician_pipeline = self._add_filters(pipeline, hashtag, start_time, end_time, user_type='politician')
        politician_df = self._perform_count_aggregation(politician_pipeline, collection)

        suspect_politician_pipeline = self._add_filters(pipeline, hashtag, start_time, end_time,
                                                        user_type='suspect_politician')
        suspect_politician_df = self._perform_count_aggregation(suspect_politician_pipeline, collection)

        df = pd.concat([normal_df, suspect_df, politician_df, suspect_politician_df], axis=1)
        df.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']

        return df

    def _get_count_area_plot(self, pipeline, collection, hashtag, start_time, end_time):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.database)
        self._validate_collection(database, collection)
        collection = database.get_collection(collection)

        df = self._get_count_data(pipeline, hashtag, start_time, end_time, collection)

        if len(df) == 1:
            plot = px.bar(df, labels={"value": "Count"})
        else:
            plot = px.area(df, labels={"value": "Count"})

        return plot

    @staticmethod
    def _add_filters(pipeline, hashtag, start_time, end_time, user_type):
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

        if hashtag:
            pipeline.insert(0, {'$match': {'entities.hashtags.tag': hashtag}})
        if start_time:
            start_time = pd.to_datetime(start_time)
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            end_time = pd.to_datetime(end_time)
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
                node = egonet.vs.find(username=user)
                egonet = egonet.induced_subgraph(egonet.neighborhood(node, order=depth))
            except (RuntimeError, ValueError) as ex:
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

        node_pipeline = [
            {'$group': {'_id': '$author.id', 'username': {'$first': '$author.username'},
                        'remiss_metadata': {'$first': '$author.remiss_metadata'}}},
        ]
        authors = collection.aggregate(node_pipeline)

        edge_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'author': '$author.id', 'referenced_by': '$referenced_tweets.author.id'}},
        ]
        edge_data = collection.aggregate(edge_pipeline)

        nodes, edges, usernames, is_suspicious, party, = {}, [], [], [], []
        for i, author in enumerate(authors):
            nodes[author['_id']] = i
            usernames.append(author['username'])
            party.append(author['remiss_metadata']['party'])
            is_suspicious.append(author['remiss_metadata']['is_usual_suspect'])

        for edge in edge_data:
            if edge['referenced_by'] not in nodes:
                nodes[edge['referenced_by']] = len(nodes)
                usernames.append(f'Unknown username: {edge["referenced_by"]}')
                is_suspicious.append(False)
                party.append(None)
            edges.append((nodes[edge['author']], nodes[edge['referenced_by']]))


        client.close()

        g = ig.Graph(directed=True)
        g.add_vertices(range(len(nodes)))
        g.vs['id'] = list(nodes.keys())
        g.vs['username'] = usernames
        g.vs['is_usual_suspect'] = is_suspicious
        g.vs['party'] = party
        g.add_edges(edges)

        return g

    def plot_egonet(self, collection, user, depth):
        network = self.get_egonet(collection, user, depth)
        return self.plot_network(network)

    def plot_network(self, network):
        layout = network.layout(self.layout)

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
