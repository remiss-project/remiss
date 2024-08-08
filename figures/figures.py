import json
import re
from abc import ABC

import plotly
import pymongoarrow.monkey
import requests
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongoarrow.schema import Schema
from pymongoarrow.monkey import patch_all

patch_all()


class MongoPlotFactory(ABC):
    def __init__(self, host="localhost", port=27017, available_datasets=None):
        super().__init__()
        self.host = host
        self.port = port
        self._available_datasets = available_datasets
        self._min_max_dates = {}
        self._available_hashtags = {}

    def _validate_dataset(self, client, dataset):
        if dataset not in client.list_database_names():
            raise RuntimeError(f'Dataset {dataset} not found')
        else:
            collections = client.get_database(dataset).list_collection_names()
            if 'raw' not in collections:
                raise RuntimeError(f'Collection raw not found in dataset {dataset}')

    def get_date_range(self, dataset):
        if dataset not in self._min_max_dates:
            client = MongoClient(self.host, self.port)
            self._validate_dataset(client, dataset)
            database = client.get_database(dataset)
            dataset = database.get_collection('raw')
            self._min_max_dates[dataset] = self._get_date_range(dataset)
            client.close()
        return self._min_max_dates[dataset]

    def get_hashtag_freqs(self, dataset):
        if dataset not in self._available_hashtags:
            client = MongoClient(self.host, self.port)
            self._validate_dataset(client, dataset)
            database = client.get_database(dataset)
            collection = database.get_collection('raw')
            self._available_hashtags[dataset] = self._get_hashtag_freqs(collection)
            client.close()
        return self._available_hashtags[dataset]

    def get_hashtags_for_user(self, dataset, user_id):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        pipeline = [
            {'$match': {'author.id': user_id}},
            {'$unwind': '$entities.hashtags'},
            {'$group': {'_id': '$entities.hashtags.tag', 'count': {'$count': {}}}},
            {'$sort': {'count': -1}}
        ]
        available_hashtags_freqs = list(collection.aggregate(pipeline))
        available_hashtags_freqs = [(x['_id'], x['count']) for x in available_hashtags_freqs]
        client.close()
        return available_hashtags_freqs

    def get_users(self, dataset):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        pipeline = [
            {'$group': {'_id': '$author.id',
                        'username': {'$first': '$author.username'}}},
            {'$project': {'_id': 0, 'username': 1, 'author_id': '$_id', }}
        ]
        schema = Schema({'username': str, 'author_id': str})
        available_users = collection.aggregate_pandas_all(pipeline, schema=schema)
        client.close()
        return available_users

    def get_user_id(self, dataset, username):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        tweet = collection.find_one({'author.username': username})
        client.close()
        if tweet:
            return tweet['author']['id']
        else:
            raise RuntimeError(f'User {username} not found')

    @property
    def available_datasets(self):
        if not self._available_datasets:
            client = MongoClient(self.host, self.port)
            available_datasets = client.list_database_names()
            client.close()
            available_datasets = list(set(available_datasets) - {'admin', 'config', 'local', 'CVCUI2'})
            if not available_datasets:
                raise RuntimeError('No datasets available')
            self._available_datasets = available_datasets
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
