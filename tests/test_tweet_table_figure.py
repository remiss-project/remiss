import uuid
from datetime import datetime
from unittest import TestCase

from pymongo import MongoClient
import pandas as pd

from figures import TweetTableFactory
import plotly.express as px


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
                      'text': 'test_text',
                      'public_metrics': {'retweet_count': 1, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_0", "id": '0',
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': False,
                                                     'has_profiling': False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]}, },
                     {"id": '1', "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      'text': 'test_text2',
                      'public_metrics': {'retweet_count': 2, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_1", "id": '1',
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': True,
                                                     'has_profiling': False
                                                     }},
                      "entities": {"hashtags": []}, },
                     {"id": '2', "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
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
        textual_data = [{'id': '0', 'fakeness_probabilities': 0.1},
                        {'id': '1', 'fakeness_probabilities': 0.2},
                        {'id': '2', 'fakeness_probabilities': 0.3}]
        textual.insert_many(textual_data)

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
                    'Suspicious content': {0: 0.1, 1: 0.2, 2: 0.3}
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[2, 1, 0]].reset_index(drop=True)
        expected = expected[['User', 'Text', 'Retweets', 'Is usual suspect', 'Party',
                             'Multimodal', 'Profiling', 'ID', 'Author ID',
                             'Fakeness']]

        pd.testing.assert_frame_equal(actual, expected)

    def test_get_top_table_full(self):
        dataset = self.test_dataset
        actual = self.top_table_factory.get_top_table_data(dataset)
        # plot plotly histogram of how many times each tweet is repeated
        # hist = actual.groupby('tweet_id').size().reset_index(name='counts')
        # hist = hist.sort_values(by='counts', ascending=False)
        # fig = px.bar(hist, x='tweet_id', y='counts')
        # fig.show()

        self.assertEqual(actual.columns.to_list(), ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party',
                                                    'Multimodal', 'Profiling', 'ID', 'Author ID', 'Fakeness'])
        client = MongoClient('localhost', 27017)
        raw = client.get_database(dataset).get_collection('raw')
        document_count = raw.count_documents({'referenced_tweets': {'$exists': False}})
        self.assertEqual(actual.shape[0], document_count)
        multimodal = client.get_database(dataset).get_collection('multimodal')
        multimodal_expected_ids = {x['tweet_id'] for x in multimodal.find({})}
        multimodal_actual_ids = set(actual[actual['Multimodal']]['ID'].to_list())
        self.assertTrue(multimodal_actual_ids.issubset(multimodal_expected_ids))
        profiling = client.get_database(dataset).get_collection('profiling')
        profiling_expected_ids = {x['user_id'] for x in profiling.find({})}
        profiling_actual_ids = set(actual[actual['Profiling']]['Author ID'].to_list())
        self.assertTrue(profiling_actual_ids.issubset(profiling_expected_ids))

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
                    'Suspicious content': {0: 0.1, 1: 0.2, 2: 0.3}
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[1, 0]].reset_index(drop=True)
        expected = expected[['User', 'Text', 'Retweets', 'Is usual suspect', 'Party',
                             'Multimodal', 'Profiling', 'ID', 'Author ID',
                             'Fakeness']]

        pd.testing.assert_frame_equal(actual, expected)

    def test_get_top_table_full_date_filtering(self):
        dataset = self.test_dataset
        actual = self.top_table_factory.get_top_table_data(dataset,
                                                           start_time=datetime.fromisoformat('2019-01-01T00:00:00Z'),
                                                           end_time=datetime.fromisoformat('2019-01-02T00:00:00Z'))
        # plot plotly histogram of how many times each tweet is repeated
        # hist = actual.groupby('tweet_id').size().reset_index(name='counts')
        # hist = hist.sort_values(by='counts', ascending=False)
        # fig = px.bar(hist, x='tweet_id', y='counts')
        # fig.show()

        self.assertEqual(actual.columns.to_list(), ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party',
                                                    'Multimodal', 'Profiling', 'ID', 'Author ID', 'Fakeness'])
        client = MongoClient('localhost', 27017)
        raw = client.get_database(dataset).get_collection('raw')
        document_count = raw.count_documents({'referenced_tweets': {'$exists': False},
                                              'created_at': {'$gte': datetime.fromisoformat('2019-01-01T00:00:00Z'),
                                                             '$lt': datetime.fromisoformat('2019-01-02T00:00:00Z')}})
        self.assertEqual(actual.shape[0], document_count)
