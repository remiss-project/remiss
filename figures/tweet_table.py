import pandas as pd
import pymongoarrow.monkey
import pymongoarrow.monkey
from pymongo import MongoClient
from pymongoarrow.schema import Schema

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()


class TweetTableFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None, limit=50):
        super().__init__(host, port, available_datasets)
        self.limit = limit
        self.top_table_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party']

    def get_top_table_data(self, dataset, start_time=None, end_time=None, only_multimodal=False, only_profiling=False):
        pipeline = [
            {'$group': {'_id': '$text', 'User': {'$first': '$author.username'},
                        'tweet_id': {'$first': '$id'},
                        'Retweets': {'$max': '$public_metrics.retweet_count'},
                        'Is usual suspect': {'$max': '$author.remiss_metadata.is_usual_suspect'},
                        'Party': {'$max': '$author.remiss_metadata.party'},
                        }},
            {'$sort': {'Retweets': -1}},
            {'$limit': self.limit},
            {'$project': {'_id': 0, 'tweet_id': 1, 'User': 1, 'Text': '$_id', 'Retweets': 1, 'Is usual suspect': 1,
                          'Party': 1}}

        ]
        pipeline = self._add_filters(pipeline, start_time, end_time, only_multimodal, only_profiling)
        schema = Schema({'tweet_id': str, 'User': str, 'Text': str, 'Retweets': int, 'Is usual suspect': bool,
                         'Party': str})
        df = self._perform_top_aggregation(pipeline, dataset, schema)
        df = df.set_index('tweet_id')

        return df

    def _add_filters(self, pipeline, start_time=None, end_time=None, only_multimodal=False, only_profiling=False):
        pipeline = pipeline.copy()
        if start_time:
            start_time = pd.to_datetime(start_time)
            pipeline.insert(0, {'$match': {'created_at': {'$gte': start_time}}})
        if end_time:
            end_time = pd.to_datetime(end_time)
            # Add a day to account for all the tweets published in that day
            end_time = end_time + pd.Timedelta(days=1)
            pipeline.insert(0, {'$match': {'created_at': {'$lte': end_time}}})
        if only_multimodal:
            pipeline.insert(0, {'$match': {'author.remiss_metadata.has_multimodal_fact-checking': True}})
        if only_profiling:
            pipeline.insert(0, {'$match': {'author.remiss_metadata.has_profiling': True}})
        return pipeline

    def _perform_top_aggregation(self, pipeline, dataset, schema=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        dataset = database.get_collection('raw')
        top_prolific = dataset.aggregate_pandas_all(pipeline, schema=schema)
        client.close()
        return top_prolific
