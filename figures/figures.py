import json
import re
from abc import ABC

import plotly
import pymongoarrow.monkey
import requests
from pymongo import MongoClient
from pymongo.errors import OperationFailure

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
            self._available_datasets = {x.capitalize().strip(): x for x in self._available_datasets}
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


class RemoteAPIFactory(ABC):
    def __init__(self, api_url, chart_id):
        super().__init__()
        self.api_url = api_url
        self.chart_id = chart_id

    def plotly_html_to_figure(self, html):
        data_str = re.findall(r'<script type="application/json" data-for="htmlwidget-.*">(.*)</script>', html)[-1]
        call_args = json.loads(f'[{data_str}]')
        with open('test_resources/plotly.json', 'w') as f:
            f.write(json.dumps(call_args))
        data = call_args[0]['x']['data']
        layout = call_args[0]['x']['layout']
        plotly_json = {'data': data, 'layout': layout}
        return plotly.io.from_json(json.dumps(plotly_json), skip_invalid=True)

    def fetch_plot_html(self, dataset=None, start_time=None, end_time=None):
        response = requests.get(f'{self.api_url}/{self.chart_id}',
                                params={'name': dataset, 'start_time': start_time, 'end_time': end_time})
        if response.status_code == 200:
            # Return plotly figure
            return response.text

        else:
            raise RuntimeError(f'Error {response.status_code} while fetching emotion line per hour data.')
