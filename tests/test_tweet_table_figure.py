import uuid
from datetime import datetime
from unittest import TestCase

import pandas as pd
from pymongo import MongoClient

from figures import TweetTableFactory


class TestTopTableFactory(TestCase):
    def setUp(self):
        self.top_table_factory = TweetTableFactory()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)
        self.client = MongoClient('localhost', 27017)
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
        textual_data = [{'id': 0, 'fakeness_probabilities': 0.1},
                        {'id': 1, 'fakeness_probabilities': 0.2},
                        {'id': 2, 'fakeness_probabilities': 0.3}]
        textual.insert_many(textual_data)

        network_metrics_collection = self.database.get_collection('network_metrics')
        network_metrics_data = [{'author_id': '0', 'legitimacy': 0, 'average_reputation': 0, 'average_status': 3},
                                {'author_id': '1', 'legitimacy': 0, 'average_reputation': 0, 'average_status': 3},
                                {'author_id': '2', 'legitimacy': 0, 'average_reputation': 0, 'average_status': 3}]
        network_metrics_collection.insert_many(network_metrics_data)

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_get_top_table(self):
        dataset = self.tmp_dataset
        actual = self.top_table_factory.get_top_table_data(dataset)
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
                    'Cascade size': {0: 1, 1: 1, 2: 1},
                    'Legitimacy': {0: 0.0, 1: 0.0, 2: 0.0},
                    'Reputation': {0: 0.0, 1: 0.0, 2: 0.0},
                    'Status': {0: 3.0, 1: 3.0, 2: 3.0},
                    'Conversation ID': {0: '0', 1: '1', 2: '2'}
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[2, 1, 0]].reset_index(drop=True)
        expected_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party', 'Multimodal', 'Profiling', 'ID',
                            'Author ID', 'Suspicious content', 'Cascade size', 'Legitimacy', 'Reputation', 'Status',
                            'Conversation ID']
        expected = expected[expected_columns]
        actual = actual[expected_columns].reset_index(drop=True)

        pd.testing.assert_frame_equal(actual, expected, check_index_type=False, check_dtype=False)

    def test_get_top_table_date_filtering(self):
        dataset = self.tmp_dataset
        actual = self.top_table_factory.get_top_table_data(dataset,
                                                           start_time=datetime.fromisoformat('2019-01-01T00:00:00Z'),
                                                           end_time=datetime.fromisoformat('2019-01-02T00:00:00Z'))
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
                    'Cascade size': {0: 1, 1: 1, 2: 1},
                    'Legitimacy': {0: 0.0, 1: 0.0, 2: 0.0},
                    'Reputation': {0: 0.0, 1: 0.0, 2: 0.0},
                    'Status': {0: 3.0, 1: 3.0, 2: 3.0},
                    'Conversation ID': {0: '0', 1: '1', 2: '2'}
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[1, 0]].reset_index(drop=True)
        expected_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party', 'Multimodal', 'Profiling', 'ID',
                            'Author ID', 'Suspicious content', 'Cascade size', 'Legitimacy', 'Reputation', 'Status',
                            'Conversation ID']
        expected = expected[expected_columns]
        actual = actual[expected_columns].reset_index(drop=True)

        pd.testing.assert_frame_equal(actual, expected, check_index_type=False, check_dtype=False)

    def test_get_top_table_full_date_filtering(self):
        dataset = self.test_dataset
        actual = self.top_table_factory.get_top_table_data(dataset,
                                                           start_time=datetime.fromisoformat('2019-01-01T00:00:00Z'),
                                                           end_time=datetime.fromisoformat('2019-01-02T00:00:00Z'))

        client = MongoClient('localhost', 27017)
        raw = client.get_database(dataset).get_collection('raw')
        document_count = raw.count_documents({'referenced_tweets': {'$exists': False},
                                              'created_at': {'$gte': datetime.fromisoformat('2019-01-01T00:00:00Z'),
                                                             '$lt': datetime.fromisoformat('2019-01-02T00:00:00Z')}})
        self.assertEqual(actual.shape[0], document_count)
