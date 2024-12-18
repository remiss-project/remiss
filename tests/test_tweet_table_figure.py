from datetime import datetime
from unittest import TestCase

import pandas as pd
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all

from figures import TweetTableFactory

patch_all()


class TestTopTableFactory(TestCase):
    def setUp(self):
        self.top_table_factory = TweetTableFactory()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.client = MongoClient('localhost', 27017)
        if self.tmp_dataset in self.client.list_database_names():
            self.client.drop_database(self.tmp_dataset)
        self.database = self.client.get_database(self.tmp_dataset)
        self.collection = self.database.get_collection('raw')
        test_data = [{"id": '0', "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                      'conversation_id': '0',
                      'text': 'test_text',
                      'public_metrics': {'retweet_count': 1, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_0", "id": '0',
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': False,
                                                     'has_profiling': False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]}, },
                     {"id": '1', "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      'conversation_id': '1',
                      'text': 'test_text2',
                      'public_metrics': {'retweet_count': 2, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_1", "id": '1',
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': True,
                                                     'has_profiling': False
                                                     }},
                      "entities": {"hashtags": []}, },
                     {"id": '2', "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      'conversation_id': '2',
                      'text': 'test_text3',
                      'public_metrics': {'retweet_count': 3, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_2", "id": '2',
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True,
                                                     }},
                      "entities": {"hashtags": []}
                      }]
        self.collection.insert_many(test_data)

        multimodal_collection = self.database.get_collection('multimodal')
        multimodal_data = [{'tweet_id': '0'}, {'tweet_id': '2'}]
        multimodal_collection.insert_many(multimodal_data)

        profiling_collection = self.database.get_collection('profiling')
        profiling_data = [{'user_id': '0'}, {'user_id': '1'}]
        profiling_collection.insert_many(profiling_data)

        textual = self.database.get_collection('textual')
        textual_data = [{'id_str': '0', 'fakeness_probabilities': 0.1},
                        {'id_str': '1', 'fakeness_probabilities': 0.2},
                        {'id_str': '2', 'fakeness_probabilities': 0.3}]
        textual.insert_many(textual_data)

        network_metrics_collection = self.database.get_collection('network_metrics')
        network_metrics_data = [
            {'author_id': '0', 'legitimacy_level': 'Low', 'reputation_level': 'High', 'status_level': 'Low'},
            {'author_id': '1', 'legitimacy_level': 'Medium', 'reputation_level': 'Medium', 'status_level': 'Low'},
            {'author_id': '2', 'legitimacy_level': 'High', 'reputation_level': 'Low', 'status_level': 'Low'}]
        network_metrics_collection.insert_many(network_metrics_data)

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_get_top_table(self):
        dataset = self.tmp_dataset
        actual = self.top_table_factory.compute_tweet_table_data(dataset)
        expected = {'Is usual suspect': {0: False, 1: False, 2: True},
                    'Party': {0: 'PSOE', 1: None, 2: 'VOX'},
                    'Retweets': {0: 1, 1: 2, 2: 3},
                    'Text': {0: 'test_text', 1: 'test_text2', 2: 'test_text3'},
                    'User': {0: 'TEST_USER_0', 1: 'TEST_USER_1', 2: 'TEST_USER_2'},
                    'Multimodal': {0: True, 1: False, 2: True},
                    'Profiling': {0: True, 1: True, 2: False},
                    'ID': {0: '0', 1: '1', 2: '2'},
                    'Author ID': {0: '0', 1: '1', 2: '2'},
                    'Suspicious content': {0: 0.1, 1: 0.2, 2: 0.3},
                    'Legitimacy': {0: 'Low', 1: 'Medium', 2: 'High'},
                    'Reputation': {0: 'High', 1: 'Medium', 2: 'Low'},
                    'Status': {0: 'Low', 1: 'Low', 2: 'Low'},
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[2, 1, 0]].reset_index(drop=True)
        expected_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party', 'Multimodal', 'Profiling', 'ID',
                            'Author ID', 'Suspicious content', 'Legitimacy', 'Reputation', 'Status']
        expected = expected[expected_columns]
        actual = actual[expected_columns].reset_index(drop=True)

        pd.testing.assert_frame_equal(actual, expected, check_index_type=False, check_dtype=False)

    def test_get_top_table_date_filtering(self):
        dataset = self.tmp_dataset
        actual = self.top_table_factory.compute_tweet_table_data(dataset,
                                                                 start_time=datetime.fromisoformat(
                                                                     '2019-01-01T00:00:00Z'),
                                                                 end_time=datetime.fromisoformat(
                                                                     '2019-01-02T00:00:00Z'))
        expected = {'Is usual suspect': {0: False, 1: False, 2: True},
                    'Party': {0: 'PSOE', 1: None, 2: 'VOX'},
                    'Retweets': {0: 1, 1: 2, 2: 3},
                    'Text': {0: 'test_text', 1: 'test_text2', 2: 'test_text3'},
                    'User': {0: 'TEST_USER_0', 1: 'TEST_USER_1', 2: 'TEST_USER_2'},
                    'Multimodal': {0: True, 1: False, 2: True},
                    'Profiling': {0: True, 1: True, 2: False},
                    'ID': {0: '0', 1: '1', 2: '2'},
                    'Author ID': {0: '0', 1: '1', 2: '2'},
                    'Suspicious content': {0: 0.1, 1: 0.2, 2: 0.3},
                    'Legitimacy': {0: 'Low', 1: 'Medium', 2: 'High'},
                    'Reputation': {0: 'High', 1: 'Medium', 2: 'Low'},
                    'Status': {0: 'Low', 1: 'Low', 2: 'Low'},
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[1, 0]].reset_index(drop=True)
        expected_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party', 'Multimodal', 'Profiling', 'ID',
                            'Author ID', 'Suspicious content', 'Legitimacy', 'Reputation', 'Status',
                            ]
        expected = expected[expected_columns]
        actual = actual[expected_columns].reset_index(drop=True)

        pd.testing.assert_frame_equal(actual, expected, check_index_type=False, check_dtype=False)

    def test_get_top_table_full_date_filtering(self):
        dataset = self.test_dataset
        actual = self.top_table_factory.compute_tweet_table_data(dataset,
                                                                 start_time=datetime.fromisoformat(
                                                                     '2019-01-01T00:00:00Z'),
                                                                 end_time=datetime.fromisoformat(
                                                                     '2019-01-02T00:00:00Z'))

        client = MongoClient('localhost', 27017)
        raw = client.get_database(dataset).get_collection('raw')
        document_count = raw.count_documents({'referenced_tweets': {'$exists': False},
                                              'created_at': {'$gte': datetime.fromisoformat('2019-01-01T00:00:00Z'),
                                                             '$lt': datetime.fromisoformat('2019-01-02T00:00:00Z')}})
        self.assertEqual(actual.shape[0], document_count)

    def test_persist(self):
        self.top_table_factory.persist([self.test_dataset])
        actual = self.top_table_factory._load_table_from_mongo(self.test_dataset)

        expected = self.top_table_factory.compute_tweet_table_data(self.test_dataset)
        expected = expected.drop(columns=['hashtags', 'created_at'])
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.test_dataset)
        collection = database.get_collection('tweet_table')
        hashtags = list(collection.find({'hashtags': {'$exists': True}}))
        self.assertGreater(len(hashtags), 0)
        created_at = list(collection.find({'created_at': {'$exists': True}}))
        self.assertGreater(len(created_at), 0)

    def test_skip(self):
        expected = self.top_table_factory._load_table_from_mongo(self.test_dataset)
        expected = expected.iloc[10:].reset_index(drop=True)
        actual = self.top_table_factory.get_tweet_table(self.test_dataset, start_tweet=10)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_limit(self):
        expected = self.top_table_factory._load_table_from_mongo(self.test_dataset)
        expected = expected.iloc[:10].reset_index(drop=True)
        actual = self.top_table_factory.get_tweet_table(self.test_dataset, amount=10)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_skip_and_limit(self):
        expected = self.top_table_factory._load_table_from_mongo(self.test_dataset)
        expected = expected.iloc[10:20].reset_index(drop=True)
        actual = self.top_table_factory.get_tweet_table(self.test_dataset, start_tweet=10, amount=10)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_hashtags(self):
        expected = self.top_table_factory._load_table_from_mongo(self.test_dataset)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.test_dataset)
        collection = database.get_collection('tweet_table')
        has_hashtag = []
        for doc in collection.find():
            try:
                if 'OpenMafia' in doc['hashtags']:
                    has_hashtag.append(doc['ID'])
            except Exception as e:
                pass
        # has_hashtag = has_hashtag[10:20]
        expected = expected[expected['ID'].isin(has_hashtag)].reset_index(drop=True)
        expected = expected.iloc[10:20].reset_index(drop=True)
        actual = self.top_table_factory.get_tweet_table(self.test_dataset, start_tweet=10, amount=10,
                                                        hashtags=['OpenMafia'])
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        self.assertGreater(len(actual), 0)

    def test_start_time(self):
        client = MongoClient('localhost', 27017)
        pipeline = [
            {'$project': {'ID': 1, 'created_at': 1}},
        ]
        collection = client.get_database(self.test_dataset).get_collection('tweet_table')
        expected = collection.aggregate_pandas_all(pipeline)
        expected['created_at'] = pd.to_datetime(expected['created_at'])
        start_time = pd.to_datetime(expected['created_at'].iloc[10].date())
        expected = expected[expected['created_at'] >= start_time].reset_index(drop=True)
        actual = self.top_table_factory.get_tweet_table(self.test_dataset, start_time=start_time)
        actual = set(actual['ID'])
        expected = set(expected['ID'])
        self.assertSetEqual(actual, expected)

    def test_end_time(self):
        client = MongoClient('localhost', 27017)
        pipeline = [
            {'$project': {'ID': 1, 'created_at': 1}},
        ]
        collection = client.get_database(self.test_dataset).get_collection('tweet_table')
        expected = collection.aggregate_pandas_all(pipeline)
        expected['created_at'] = pd.to_datetime(expected['created_at'])
        end_time = expected['created_at'].iloc[10] + pd.Timedelta(days=1)
        expected = expected[expected['created_at'] < end_time].reset_index(drop=True)
        actual = self.top_table_factory.get_tweet_table(self.test_dataset, end_time=end_time)
        actual = set(actual['ID'])
        expected = set(expected['ID'])
        self.assertTrue(len(expected - actual) < 5)

    # def test_fix_numerical_ids(self):
    #     datasets = ['MENA_Agressions', 'MENA_Ajudes', 'Barcelona_2019', 'Generalitat_2021']
    #     client = MongoClient('mongodb://srvinv02.esade.es', 27017)
    #     for dataset in datasets:
    #
    #         database = client.get_database(dataset)
    #         textual = database.get_collection('textual')
    #         # Store ids as string in id_str as a regular field
    #         pipeline = [
    #             # Add id_str field if it does not exist
    #             {'$addFields': {'id_str': {'$toString': '$id'}}},
    #         ]
    #         textual.update_many({'id_str':{'$exists': False}}, pipeline)
    #
    #     client.close()

    def test_all_fields_present(self):
        self.top_table_factory.host = 'mongodb://srvinv02.esade.es'
        datasets = [
            # 'Openarms', 'MENA_Agressions', 'MENA_Ajudes', 'Barcelona_2019',
            'Andalucia_2022',
            # 'Generales_2019',
            # 'Generalitat_2021',
        ]
        for dataset in datasets:
            print(f'Plotting {dataset}')
            actual = self.top_table_factory.get_tweet_table(dataset, start_tweet=0, amount=50)
            self.assertGreater(len(actual), 0)
            self.assertEqual(
                set(actual.columns.to_list()), {'ID', 'User', 'Text', 'Retweets', 'Is usual suspect', 'Party',
                                                'Multimodal', 'Profiling', 'Suspicious content', 'Legitimacy',
                                                'Reputation', 'Status', 'Author ID'})
            self.assertFalse(actual['Suspicious content'].isna().all())
