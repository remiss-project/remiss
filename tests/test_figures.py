import unittest
from unittest.mock import Mock, patch

from pymongo import MongoClient

from figures.figures import MongoPlotFactory


class TestMongoPlotFactory(unittest.TestCase):
    def setUp(self):
        self.mongo_plot = MongoPlotFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_remiss')
        self.database = self.client.get_database('test_remiss')
        self.collection = self.database.get_collection('test_collection')
        test_data = [{"id": 0, "created_at": {"$date": "2019-01-01T23:20:00Z"},
                      "author": {"username": "TEST_USER_0", "id": 0,
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}, {"tag": "test_hashtag2"}]},
                      "referenced_tweets": []},
                     {"id": 1, "created_at": {"$date": "2019-01-02T23:20:00Z"},
                      "author": {"username": "TEST_USER_1", "id": 1,
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}, {"tag": "test_hashtag2"}]},
                      "referenced_tweets": [{"id": 1, "type": "quoted"}]},
                     {"id": 2, "created_at": {"$date": "2019-01-03T23:20:00Z"},
                      "author": {"username": "TEST_USER_2", "id": 2,
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": [{"id": 1, "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

    def tearDown(self) -> None:
        self.collection.drop()
        self.client.drop_database('test_remiss')
        self.client.close()

    def test_get_date_range(self):
        # Mock MongoClient and database
        mock_date = Mock()
        mock_date.date.return_value = 'test_date'
        mock_collection = Mock()
        mock_collection.find_one.return_value = {'created_at': mock_date}
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        with patch('figures.figures.MongoClient', return_value=mock_client):
            date_range = self.mongo_plot.get_date_range("test_collection")
            self.assertEqual(date_range, ('test_date', 'test_date'))

    def test_get_hashtag_freqs(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_collection.aggregate.return_value = [
            {'_id': 'test_hashtag1', 'count': 1},
            {'_id': 'test_hashtag2', 'count': 2},
            {'_id': 'test_hashtag3', 'count': 3},
        ]
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        with patch('figures.figures.MongoClient', return_value=mock_client):
            hashtag_freqs = self.mongo_plot.get_hashtag_freqs("test_collection")
            self.assertEqual(hashtag_freqs, [(x['_id'], x['count']) for x in mock_collection.aggregate.return_value])

    def test_get_hashtag_freqs_2(self):
        expected = [('test_hashtag', 3), ('test_hashtag2', 2)]
        actual = self.mongo_plot.get_hashtag_freqs("test_collection")
        self.assertEqual(expected, actual)

    def test_get_users(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_collection.distinct.return_value = ['test_user1', 'test_user2']
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        with patch('figures.figures.MongoClient', return_value=mock_client):
            users = self.mongo_plot.get_users("test_collection")
            self.assertEqual(users, [str(x) for x in mock_collection.distinct.return_value])

    def test_get_users_2(self):
        expected = ['TEST_USER_0', 'TEST_USER_1', 'TEST_USER_2']
        actual = self.mongo_plot.get_users("test_collection")
        self.assertEqual(expected, actual)

    def test_get_user_id(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database

        # Mock find_one
        mock_collection.find_one.return_value = {
            'author': {'username': 'test_username', 'id': 'test_id'}
        }

        with patch('figures.figures.MongoClient', return_value=mock_client):
            user_id = self.mongo_plot.get_user_id("test_collection", "test_username")
            self.assertEqual(user_id, 'test_id')

    def test_get_user_id_2(self):
        expected = 0
        actual = self.mongo_plot.get_user_id("test_collection", "TEST_USER_0")
        self.assertEqual(expected, actual)

    def test_available_datasets(self):
        # Mock MongoClient
        mock_client = Mock()
        mock_client.get_database.return_value.list_collection_names.return_value = ['collection1', 'collection2']
        with patch('figures.figures.MongoClient', return_value=mock_client):
            datasets = self.mongo_plot.available_datasets
            self.assertEqual(datasets, mock_client.get_database.return_value.list_collection_names.return_value)

    def test_available_datasets_2(self):
        expected = ['test_collection']
        actual = self.mongo_plot.available_datasets
        self.assertEqual(expected, actual)

    def test__get_hashtag_freqs(self):
        expected = [('test_hashtag', 3), ('test_hashtag2', 2)]
        self.mongo_plot.max_hashtags = None
        actual = self.mongo_plot._get_hashtag_freqs(self.collection)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
