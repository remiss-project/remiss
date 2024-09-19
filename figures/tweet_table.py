import pandas as pd
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from pymongoarrow.schema import Schema

from figures.figures import MongoPlotFactory

patch_all()


class TweetTableFactory(MongoPlotFactory):

    def get_top_table_data(self, dataset, start_time=None, end_time=None, hashtags=None):

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
            {'$project': {
                '_id': 0,
                'username': '$author.username',
                'author_id': '$author.id',
                'tweet_id': '$id',
                'text': '$text',
                'retweets': '$public_metrics.retweet_count',
                'suspect': '$author.remiss_metadata.is_usual_suspect',
                'party': '$author.remiss_metadata.party',
                'conversation_id': '$conversation_id', }},
            {'$addFields': {'tweet_id_int': {'$toLong': '$tweet_id'}}},
        ]
        initial_schema = Schema({
            'username': str,
            'author_id': str,
            'tweet_id': str,
            'text': str,
            'retweets': int,
            'suspect': bool,
            'party': str,
            'conversation_id': str,
            'tweet_id_int': int
        })
        pipeline_initial = self._add_filters(pipeline_initial, start_time, end_time, hashtags)
        df_initial = collection_raw.aggregate_pandas_all(pipeline_initial, schema=initial_schema)

        # Multimodal Collection
        pipeline_multimodal = [
            {'$project': {'_id': 0, 'tweet_id': 1, 'Multimodal': {'$literal': True}}}
        ]
        multimodal_schema = Schema({
            'tweet_id': str,
            'Multimodal': bool
        })
        df_multimodal = collection_multimodal.aggregate_pandas_all(pipeline_multimodal, schema=multimodal_schema)

        # Profiling Collection
        pipeline_profiling = [
            {'$project': {'_id': 0, 'user_id': '$user_id', 'Profiling': {'$literal': True}}}
        ]
        profiling_schema = Schema({
            'user_id': str,
            'Profiling': bool
        })
        df_profiling = collection_profiling.aggregate_pandas_all(pipeline_profiling, schema=profiling_schema)

        # Textual Collection
        pipeline_textual = [
            {'$project': {'_id': 0, 'id': 1, 'Suspicious content': '$fakeness_probabilities'}}
        ]
        textual_schema = Schema({
            'id': int,
            'Suspicious content': float
        })
        df_textual = collection_textual.aggregate_pandas_all(pipeline_textual, schema=textual_schema)

        # Conversation Size from Raw Collection
        pipeline_conversation = [
            {'$group': {'_id': '$conversation_id', 'Cascade size': {'$sum': 1}}},
            {'$project': {'_id': 0, 'conversation_id': '$_id', 'Cascade size': 1}}
        ]
        conversation_schema = Schema({
            'conversation_id': str,
            'Cascade size': int
        })
        df_conversation = collection_raw.aggregate_pandas_all(pipeline_conversation, schema=conversation_schema)

        # Network Metrics Collection
        pipeline_network_metrics = [
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
        df_final = df_final.merge(df_textual, left_on='tweet_id_int', right_on='id', how='left')
        df_final = df_final.merge(df_conversation, on='conversation_id', how='left')
        df_final = df_final.merge(df_network_metrics, on='author_id', how='left')

        df_final = df_final.drop(columns=['tweet_id_int', 'user_id', 'id'])
        # Fill missing values and sort
        with pd.option_context("future.no_silent_downcasting", True):
            df_final['Multimodal'] = df_final['Multimodal'].fillna(False).infer_objects(copy=False)
            df_final['Profiling'] = df_final['Profiling'].fillna(False).infer_objects(copy=False)
        df_final.sort_values(by='retweets', ascending=False, inplace=True)

        df_final = df_final.rename(columns={'username': 'User', 'text': 'Text', 'retweets': 'Retweets',
                                            'suspect': 'Is usual suspect', 'party': 'Party', 'tweet_id': 'ID',
                                            'author_id': 'Author ID', 'fakeness_probabilities': 'Suspicious content',
                                            'conversation_id': 'Conversation ID', 'legitimacy': 'Legitimacy',
                                            'average_reputation': 'Reputation', 'average_status': 'Status'})

        # Close client connection
        client.close()
        return df_final


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
