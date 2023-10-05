from datetime import datetime

from dash_oop_components import DashFigureFactory
import plotly.express as px
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongoarrow.schema import Schema
import pymongoarrow.monkey

pymongoarrow.monkey.patch_all()


class CountAreaPlots(DashFigureFactory):
    def __init__(self, host="localhost", port=27017, database="test_remiss"):
        super().__init__()
        self.host = host
        self.port = port
        self.database = database
        self._available_datasets = None
        self._min_max_dates = {}
        self._available_hashtags = {}

    def plot_tweet_series(self, collection, hashtag, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]
        return self._get_counts(pipeline, collection, hashtag, start_time, end_time, unit, bin_size)

    def plot_user_series(self, collection, hashtag, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "users": {'$addToSet': "$author.username"}}
            },
            {'$project': {'count': {'$size': '$users'}}},
            {'$sort': {'_id': 1}}
        ]
        return self._get_counts(pipeline, collection, hashtag, start_time, end_time, unit, bin_size)

    def _get_counts(self, pipeline, collection, hashtag, start_time, end_time):
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

    @staticmethod
    def _add_filters(pipeline, hashtag, start_time, end_time):
        if hashtag:
            pipeline.insert(0, {'$match': {'entities.hashtags.tag': hashtag}})
        if start_time:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            pipeline.insert(0, {'$match': {'created_at': {'$lte': end_time}}})
        return pipeline

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
