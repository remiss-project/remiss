from datetime import datetime
from unittest import TestCase

from pymongo import MongoClient
import pandas as pd

from figures import TweetTableFactory


class TestTopTableFactory(TestCase):
    def setUp(self):
        self.top_table_factory = TweetTableFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_remiss')
        self.database = self.client.get_database('test_remiss')
        self.collection = self.database.get_collection('test_collection')
        test_data = [{"id": '0', "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                      'text': 'test_text',
                      'public_metrics': {'retweet_count': 1, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_0", "id": '0',
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": '1', "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      'text': 'test_text2',
                      'public_metrics': {'retweet_count': 2, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_1", "id": 1,
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": '1', "type": "quoted"}]},
                     {"id": '2', "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      'text': 'test_text3',
                      'public_metrics': {'retweet_count': 3, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_2", "id": '2',
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": '1', "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

    def test_get_top_table(self):
        test_collection = 'test_collection'
        actual = self.top_table_factory.get_top_table_data(test_collection)
        expected = {'Is usual suspect': {0: False, 1: False, 2: True},
                    'Party': {0: 'PSOE', 1: None, 2: 'VOX'},
                    'Retweets': {0: 1, 1: 2, 2: 3},
                    'Text': {0: 'test_text', 1: 'test_text2', 2: 'test_text3'},
                    'User': {0: 'TEST_USER_0', 1: 'TEST_USER_1', 2: 'TEST_USER_2'}}
        expected = pd.DataFrame(expected)
        expected.index = expected.index.astype(str)
        expected.index.name = 'tweet_id'
        expected = expected[['User', 'Text', 'Retweets', 'Is usual suspect', 'Party']]
        expected = expected.iloc[[2, 1, 0]]
        pd.testing.assert_frame_equal(actual, expected)
