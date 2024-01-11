import pandas as pd
import pymongoarrow.monkey
import pymongoarrow.monkey
from pymongo import MongoClient

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()


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
