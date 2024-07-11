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
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": '1', "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      'text': 'test_text2',
                      'public_metrics': {'retweet_count': 2, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_1", "id": '1',
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': True,
                                                     'has_profiling': False
                                                     }},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": '1', "type": "quoted"}]},
                     {"id": '2', "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      'text': 'test_text3',
                      'public_metrics': {'retweet_count': 3, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_2", "id": '2',
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True,
                                                     }},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": '1', "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

        multimodal_collection = self.database.get_collection('multimodal')
        multimodal_data = [{'tweet_id': '0'}, {'tweet_id': '2'}]
        multimodal_collection.insert_many(multimodal_data)

        profiling_collection = self.database.get_collection('profiling')
        profiling_data = [{'user_id': '0'}, {'user_id': '1'}]
        profiling_collection.insert_many(profiling_data)

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
                    'tweet_id': {0: '0', 1: '1', 2: '2'},
                    'author_id': {0: '0', 1: '1', 2: '2'}
                    }
        expected = pd.DataFrame(expected)
        expected = expected.iloc[[2, 1, 0]].reset_index(drop=True)
        expected = expected[['User', 'Text', 'Retweets', 'Is usual suspect', 'Party',
                             'Multimodal', 'Profiling', 'tweet_id', 'author_id']]

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
                                                    'Multimodal', 'Profiling', 'tweet_id', 'author_id'])
        client = MongoClient('localhost', 27017)
        raw = client.get_database(dataset).get_collection('raw')
        document_count = raw.count_documents({})
        self.assertEqual(actual.shape[0], document_count)
        multimodal = client.get_database(dataset).get_collection('multimodal')
        multimodal_expected_ids = {x['tweet_id'] for x in multimodal.find({})}
        multimodal_actual_ids = set(actual[actual['Multimodal']]['tweet_id'].to_list())
        self.assertTrue(multimodal_expected_ids.issubset(multimodal_actual_ids))
        profiling = client.get_database(dataset).get_collection('profiling')
        profiling_expected_ids = {x['user_id'] for x in profiling.find({})}
        profiling_actual_ids = set(actual[actual['Profiling']]['author_id'].to_list())
        self.assertTrue(profiling_expected_ids.issubset(profiling_actual_ids))
