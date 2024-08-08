import logging
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import pymongoarrow.monkey
from pymongo import MongoClient
from pymongoarrow.schema import Schema

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()

logger = logging.getLogger('time_series')


class TimeSeriesFactory(MongoPlotFactory):

    def compute_tweet_histogram(self, dataset, hashtags, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]

        df = self._get_count_data(dataset, pipeline, hashtags, start_time, end_time)
        return df

    def compute_user_histogram(self, dataset, hashtags, start_time, end_time, unit='day', bin_size=1):
        pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "users": {'$addToSet': "$author.username"}}
            },
            {'$project': {'count': {'$size': '$users'}}},
            {'$sort': {'_id': 1}}
        ]

        df = self._get_count_data(dataset, pipeline, hashtags, start_time, end_time)
        return df

    def plot_tweet_series(self, dataset, hashtags, start_time, end_time, unit='day', bin_size=1):
        if hashtags or start_time or end_time:
            logger.debug('Computing tweet series')
            start_computing_time = time.time()
            df = self.compute_tweet_histogram(dataset, hashtags, start_time, end_time, unit, bin_size)
            logger.debug(f'Tweet series computed in {time.time() - start_computing_time} seconds')
        else:
            df = self.load_histogram(dataset, 'tweet')

        plot = self._get_count_plot(df)
        return plot

    def plot_user_series(self, dataset, hashtags, start_time, end_time, unit='day', bin_size=1):
        if hashtags or start_time or end_time:
            logger.debug('Computing user series')
            start_computing_time = time.time()
            df = self.compute_user_histogram(dataset, hashtags, start_time, end_time, unit, bin_size)
            logger.debug(f'User series computed in {time.time() - start_computing_time} seconds')
        else:
            df = self.load_histogram(dataset, 'user')

        plot = self._get_count_plot(df)
        return plot

    def _perform_count_aggregation(self, pipeline, collection):
        df = collection.aggregate_pandas_all(
            pipeline,
            schema=Schema({'_id': datetime, 'count': int})
        )
        df = df.rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')

        return df

    def _get_count_data(self, dataset, pipeline, hashtags, start_time, end_time):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
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
        df = df.fillna(0)

        return df

    def _get_count_plot(self, df):
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

    def persist(self, datasets):
        for dataset in datasets:
            tweet_df = self.compute_tweet_histogram(dataset, [], None, None)
            user_df = self.compute_user_histogram(dataset, [], None, None)

            self._persist_histogram(tweet_df, dataset, 'tweet')
            self._persist_histogram(user_df, dataset, 'user')

    def _persist_histogram(self, df, dataset, kind):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        database.drop_collection(f'{kind}_time_series')
        collection = database.get_collection(f'{kind}_time_series')

        collection.insert_many(df.reset_index().to_dict('records'))

    def load_histogram(self, dataset, kind):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection(f'{kind}_time_series')

        schema = Schema({'Time': datetime, 'Normal': int, 'Usual suspect': int, 'Politician': int,
                         'Usual suspect politician': int})
        df = collection.aggregate_pandas_all([], schema=schema)
        df = df.set_index('Time')
        return df
