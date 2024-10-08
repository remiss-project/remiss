import logging
from datetime import datetime

import pandas as pd
from bson import Regex
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from pymongoarrow.schema import Schema

from figures.figures import MongoPlotFactory

patch_all()

logger = logging.getLogger(__name__)


class TweetTableFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None):
        super().__init__(host, port, available_datasets)
        self.operators = [['ge ', '>='],
                          ['le ', '<='],
                          ['lt ', '<'],
                          ['gt ', '>'],
                          ['ne ', '!='],
                          ['eq ', '='],
                          ['contains '],
                          ['datestartswith ']]

    def get_tweet_table(self, dataset, start_time=None, end_time=None, hashtags=None, start_tweet=None, amount=None,
                        sort_by=None, filter_query=None):
        start_time, end_time = self._validate_dates(dataset, start_time, end_time)
        pipeline = []
        pipeline = self._add_filters(pipeline, start_time, end_time, hashtags, filter_query)
        pipeline = self._add_sorting(pipeline, sort_by)
        pipeline.append(
            {'$project':
                {'_id': 0, 'User': 1, 'Text': 1, 'Retweets': 1, 'Is usual suspect': 1, 'Party': 1, 'Multimodal': 1,
                'Profiling': 1, 'ID': 1, 'Author ID': 1, 'Suspicious content': 1, 'Legitimacy': 1, 'Reputation': 1,
                'Status': 1, }})
        pipeline = self._add_pagination(pipeline, start_tweet, amount)


        schema = Schema({
            'ID': str,
            'User': str,
            'Text': str,
            'Retweets': int,
            'Is usual suspect': bool,
            'Party': str,
            'Multimodal': bool,
            'Profiling': bool,
            'Suspicious content': float,
            'Legitimacy': str,
            'Reputation': str,
            'Status': str,
            'Author ID': str,
        })
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        if 'tweet_table' not in database.list_collection_names():
            raise RuntimeError('Tweet table collection not found in the database. Please prepopulate first.')
        collection = database.get_collection('tweet_table')
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        client.close()
        return df

    def _add_filters(self, pipeline, start_time=None, end_time=None, hashtags=None, filter_query=None):
        pipeline = pipeline.copy()
        if start_time:
            start_time = pd.to_datetime(start_time)
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            end_time = pd.to_datetime(end_time)
            # Add a day to account for all the tweets published in that day
            end_time = end_time + pd.Timedelta(days=1)
            pipeline.insert(0, {'$match': {'created_at': {'$lt': end_time}}})
        if hashtags:
            # Match tweets that contain any of the hashtags in the field Hashtags of the db
            pipeline.insert(0, {'$match': {'hashtags': {'$in': hashtags}}})


        if filter_query:
            mongo_filter = self.build_mongo_filter(filter_query)
            pipeline.insert(0, {'$match': mongo_filter})

        return pipeline

    def split_filter_part(self, filter_part):
        """
        Splits a filter string into column name, operator, and value.
        Example filter: {column_name} ge 30
        """
        for operator_type in self.operators:
            for operator in operator_type:
                if operator in filter_part:
                    name_part, value_part = filter_part.split(operator, 1)
                    # Extract the column name inside {}
                    name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                    # Strip and clean up the value
                    value_part = value_part.strip()
                    v0 = value_part[0]
                    if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                        value = value_part[1: -1].replace('\\' + v0, v0)
                    else:
                        try:
                            value = float(value_part)
                        except ValueError:
                            value = value_part

                    # Return the name, operator, and cleaned value
                    return name, operator_type[0].strip(), value

        return [None] * 3

    def build_mongo_filter(self, filter_query):
        """
        Builds the MongoDB filter based on the parsed filter query.
        """
        mongo_filters = []

        if filter_query:
            # Split the query by '&&' to handle multiple conditions
            filtering_expressions = filter_query.split(' && ')
            for filter_part in filtering_expressions:
                col_name, operator, filter_value = self.split_filter_part(filter_part)

                if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                    # MongoDB comparison operators
                    mongo_filters.append({
                        col_name: {
                            self.operators[self._get_operator_index(operator)][1]: self._convert_value_for_mongo(
                                filter_value)}
                    })

                elif operator == 'contains':
                    # MongoDB regex for substring matching (case insensitive)
                    mongo_filters.append({
                        col_name: {self.operators[self._get_operator_index(operator)][0]: Regex(filter_value, 'i')}
                    })

                elif operator == 'datestartswith':
                    # MongoDB regex for "startswith" (case insensitive)
                    mongo_filters.append({
                        col_name: {
                            self.operators[self._get_operator_index(operator)][0]: Regex(f'^{filter_value}', 'i')}
                    })

        # Combine all filters using `$and`
        if mongo_filters:
            return {'$and': mongo_filters}
        else:
            return {}

    def _get_operator_index(self, operator):
        """
        Find the index of the operator type in the `self.operators` list.
        """
        for i, operator_type in enumerate(self.operators):
            if operator == operator_type[0].strip():
                return i
        return None

    def _convert_value_for_mongo(self, value):
        """
        Converts the filter value to the correct type (e.g., numeric types) for MongoDB.
        """
        try:
            # Try converting the value to float/int for numeric comparisons
            if '.' in str(value):
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If it's not a number, return the value as a string
            return value

    def _add_sorting(self, pipeline, sort_by):
        if sort_by:
            # Construct the sort dictionary
            sort_dict = {}
            for col in sort_by:
                column_id = col['column_id']  # Column to sort by
                direction = 1 if col['direction'] == 'asc' else -1  # Ascending or descending
                sort_dict[column_id] = direction
            pipeline.append({'$sort': sort_dict})

        return pipeline

    def _add_pagination(self, pipeline, start_tweet=None, amount=None):
        if start_tweet is not None:
            pipeline.append({'$skip': start_tweet})

        if amount is not None:
            pipeline.append({'$limit': amount})
        return pipeline

    def get_tweet_table_size(self, dataset, start_time=None, end_time=None, hashtags=None):
        start_time, end_time = self._validate_dates(dataset, start_time, end_time)
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('tweet_table')

        if start_time or end_time or hashtags:
            pipeline = []
            pipeline = self._add_filters(pipeline, start_time, end_time, hashtags)
            pipeline.append({'$count': 'size'})
            size = collection.aggregate(pipeline)
            if size is not None:
                size = list(size)
                if size:
                    size = size[0]['size']
                else:
                    size = 0
        else:
            size = collection.count_documents({})
        client.close()
        return size

    def compute_tweet_table_data(self, dataset, start_time=None, end_time=None, hashtags=None):

        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)

        # Collections
        collection_raw = database.get_collection('raw')
        collection_multimodal = database.get_collection('multimodal')
        collection_profiling = database.get_collection('profiling')
        collection_textual = database.get_collection('textual')
        collection_network_metrics = database.get_collection('network_metrics')

        # Initial pipeline without lookups
        pipeline_initial = [
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$project': {
                '_id': 0,
                'username': '$author.username',
                'author_id': '$author.id',
                'tweet_id': '$id',
                'text': '$text',
                'retweets': '$public_metrics.retweet_count',
                'suspect': '$author.remiss_metadata.is_usual_suspect',
                'party': '$author.remiss_metadata.party',
                'created_at': 1,
            }},
        ]
        initial_schema = Schema({
            'username': str,
            'author_id': str,
            'tweet_id': str,
            'text': str,
            'retweets': int,
            'suspect': bool,
            'party': str,
            'created_at': datetime,

        })
        pipeline_initial = self._add_filters(pipeline_initial, start_time, end_time, hashtags)
        df_initial = collection_raw.aggregate_pandas_all(pipeline_initial, schema=initial_schema)
        df_initial['hashtags'] = df_initial['text'].str.extractall(r'#(\w+)')[0].groupby(level=0).apply(list)

        # Multimodal Collection
        pipeline_multimodal = [
            {'$match': {'tweet_id': {'$in': df_initial['tweet_id'].tolist()}}},
            {'$project': {'_id': 0, 'tweet_id': 1, 'Multimodal': {'$literal': True}}}
        ]
        multimodal_schema = Schema({
            'tweet_id': str,
            'Multimodal': bool
        })
        df_multimodal = collection_multimodal.aggregate_pandas_all(pipeline_multimodal, schema=multimodal_schema)

        # Profiling Collection
        pipeline_profiling = [
            {'$match': {'user_id': {'$in': df_initial['author_id'].tolist()}}},
            {'$project': {'_id': 0, 'user_id': '$user_id', 'Profiling': {'$literal': True}}}
        ]
        profiling_schema = Schema({
            'user_id': str,
            'Profiling': bool
        })
        df_profiling = collection_profiling.aggregate_pandas_all(pipeline_profiling, schema=profiling_schema)

        # Textual Collection
        pipeline_textual = [
            {'$match': {'id_str': {'$in': df_initial['tweet_id'].tolist()}}},
            {'$project': {'_id': 0, 'id': '$id_str', 'Suspicious content': '$fakeness_probabilities'}}
        ]
        textual_schema = Schema({
            'id': str,
            'Suspicious content': float
        })
        df_textual = collection_textual.aggregate_pandas_all(pipeline_textual, schema=textual_schema)

        # Network Metrics Collection
        pipeline_network_metrics = [
            {'$match': {'author_id': {'$in': df_initial['author_id'].tolist()}}},
            {'$project': {
                '_id': 0,
                'author_id': 1,
                'Legitimacy': '$legitimacy_level',
                'Reputation': '$reputation_level',
                'Status': '$status_level'}}
        ]
        network_metrics_schema = Schema({
            'author_id': str,
            'Legitimacy': str,
            'Reputation': str,
            'Status': str
        })
        df_network_metrics = collection_network_metrics.aggregate_pandas_all(pipeline_network_metrics,
                                                                             schema=network_metrics_schema)

        # Merge DataFrames
        df_final = df_initial.merge(df_multimodal, on='tweet_id', how='left')
        df_final = df_final.merge(df_profiling, left_on='author_id', right_on='user_id', how='left')
        df_final = df_final.merge(df_textual, left_on='tweet_id', right_on='id', how='left')
        df_final = df_final.merge(df_network_metrics, on='author_id', how='left')

        df_final = df_final.drop(columns=['user_id', 'id'])
        # Fill missing values and sort
        with pd.option_context("future.no_silent_downcasting", True):
            df_final['Multimodal'] = df_final['Multimodal'].fillna(False).infer_objects(copy=False)
            df_final['Profiling'] = df_final['Profiling'].fillna(False).infer_objects(copy=False)
        df_final.sort_values(by='retweets', ascending=False, inplace=True)

        df_final = df_final.rename(columns={'username': 'User', 'text': 'Text', 'retweets': 'Retweets',
                                            'suspect': 'Is usual suspect', 'party': 'Party', 'tweet_id': 'ID',
                                            'author_id': 'Author ID', 'fakeness_probabilities': 'Suspicious content',
                                            'legitimacy': 'Legitimacy',
                                            'average_reputation': 'Reputation', 'average_status': 'Status'})

        df_final = df_final.reset_index(drop=True)
        df_final = df_final[['ID', 'User', 'Text', 'Retweets', 'Is usual suspect', 'Party', 'Multimodal', 'Profiling',
                             'Suspicious content', 'Legitimacy', 'Reputation', 'Status', 'Author ID',
                             'hashtags', 'created_at']]
        # Close client connection
        client.close()
        return df_final

    def persist(self, datasets):
        for dataset in datasets:
            logger.info(f'Processing dataset {dataset}')
            df = self.compute_tweet_table_data(dataset)
            client = MongoClient(self.host, self.port)
            database = client.get_database(dataset)
            collection = database.get_collection('tweet_table')
            collection.drop()
            collection.insert_many(df.to_dict('records'))
            client.close()

    def _load_table_from_mongo(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('tweet_table')
        pipeline = [
            {'$project': {
                '_id': 0,
                'User': 1,
                'Text': 1,
                'Retweets': 1,
                'Is usual suspect': 1,
                'Party': 1,
                'Multimodal': 1,
                'Profiling': 1,
                'ID': 1,
                'Author ID': 1,
                'Suspicious content': 1,
                'Legitimacy': 1,
                'Reputation': 1,
                'Status': 1,
            }}
        ]
        schema = Schema({
            'ID': str,
            'User': str,
            'Text': str,
            'Retweets': int,
            'Is usual suspect': bool,
            'Party': str,
            'Multimodal': bool,
            'Profiling': bool,
            'Suspicious content': float,
            'Legitimacy': str,
            'Reputation': str,
            'Status': str,
            'Author ID': str,
        })
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        return df
