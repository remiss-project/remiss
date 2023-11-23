import unittest
from datetime import datetime

from dash import Dash
from pymongo import MongoClient

from figures import TopTableFactory


class TopTableTestCase(unittest.TestCase):
    def setUp(self):
        self.top_table = TopTableFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_remiss')
        self.database = self.client.get_database('test_remiss')
        self.collection = self.database.get_collection('test_collection')
        test_data = [{"id": 0, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                      'text': 'test_text1',
                      "author": {"username": "TEST_USER_0", "id": 0,
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": 1, "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      'text': 'test_text2',
                      "author": {"username": "TEST_USER_1", "id": 1,
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "quoted"}]},
                     {"id": 2, "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      'text': 'test_text3',
                      "author": {"username": "TEST_USER_2", "id": 2,
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

    def test_top_tweets_table(self):
        table = self.top_table.get_top_tweets('test_collection')
        self.assertEqual(table.data,
                         [{'count': 1, 'id': 0, 'text': 'test_text1'},
                          {'count': 1, 'id': 1, 'text': 'test_text2'},
                          {'count': 1, 'id': 2, 'text': 'test_text3'}])

    def test_top_users_table(self):
        table = self.top_table.get_top_users('test_collection')
        self.assertEqual([{'count': 1, 'username': 'TEST_USER_1'}, {'count': 1, 'username': 'TEST_USER_2'},
                          {'count': 1, 'username': 'TEST_USER_0'}], table.data)


if __name__ == '__main__':
    unittest.main()
