import pandas as pd
from pymongoarrow.monkey import patch_all
from pymongo import MongoClient
from pymongoarrow.schema import Schema

from figures.figures import MongoPlotFactory

patch_all()


class TweetTableFactory(MongoPlotFactory):

    def get_top_table_data(self, dataset, start_time=None, end_time=None, hashtags=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        pipeline = [
            # Exclude RTs
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$project': {'username': '$author.username',
                          'author_id': '$author.id',
                          'tweet_id': '$id',
                          'text': '$text',
                          'retweets': '$public_metrics.retweet_count',
                          'suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          'conversation_id': '$conversation_id'}},
            {'$addFields': {'tweet_id_int': {'$toLong': '$tweet_id'}}},
            {'$lookup': {'from': 'multimodal', 'localField': 'tweet_id', 'foreignField': 'tweet_id',
                         'as': 'multimodal'}},
            {'$lookup': {'from': 'profiling', 'localField': 'author_id', 'foreignField': 'user_id',
                         'as': 'profiling'}},
            {'$lookup': {'from': 'textual', 'localField': 'tweet_id_int', 'foreignField': 'id',
                         'as': 'textual'}},
            {'$lookup': {'from': 'raw', 'localField': 'conversation_id', 'foreignField': 'conversation_id',
                         'as': 'conversation'}},
            {'$lookup': {'from': 'network_metrics', 'localField': 'author_id', 'foreignField': 'author_id',
                         'as': 'network_metrics'}},
            {'$project': {'User': '$username', 'Text': '$text', 'Retweets': '$retweets',
                          'Is usual suspect': '$suspect', 'Party': '$party',
                          'Multimodal': {'$cond': {'if': {'$eq': [{'$size': '$multimodal'}, 0]}, 'then': False,
                                                   'else': True}},
                          'Profiling': {'$cond': {'if': {'$eq': [{'$size': '$profiling'}, 0]}, 'then': False,
                                                  'else': True}},
                          'ID': '$tweet_id', 'Author ID': '$author_id',
                          'Suspicious content': {'$arrayElemAt': ['$textual.fakeness_probabilities', 0]},
                          'Cascade size': {'$size': '$conversation'},
                          'Legitimacy': {'$arrayElemAt': ['$network_metrics.legitimacy', 0]},
                          'Reputation': {'$arrayElemAt': ['$network_metrics.average_reputation', 0]},
                          'Status': {'$arrayElemAt': ['$network_metrics.average_status', 0]}},
             },

            {'$sort': {'Retweets': -1}},

        ]
        pipeline = self._add_filters(pipeline, start_time, end_time, hashtags)
        schema = Schema({
            'User': str,
            'Text': str,
            'Retweets': int,
            'Is usual suspect': bool,
            'Party': str,
            'Multimodal': bool,
            'Profiling': bool,
            'ID': str,
            'Author ID': str,
            'Suspicious content': float,
            'Cascade size': int,
            'Legitimacy': float,
            'Reputation': float,
            'Status': float
        })
        # df_list = list(collection.aggregate(pipeline))
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        client.close()

        return df

    def _add_filters(self, pipeline, start_time=None, end_time=None, hashtags=None):
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
            pipeline.insert(0, {'$match': {'entities.hashtags.tag': {'$in': hashtags}}})

        return pipeline
