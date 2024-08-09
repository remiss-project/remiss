import unittest
from unittest.mock import Mock, patch

from pymongo import MongoClient

from figures.figures import MongoPlotFactory


class TestMongoPlotFactory(unittest.TestCase):
    def setUp(self):
        self.mongo_plot = MongoPlotFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_dataset')
        self.database = self.client.get_database('test_dataset')
        self.collection = self.database.get_collection('raw')
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
        self.client.drop_database('test_dataset')
        self.client.close()

    def test_get_date_range(self):
        # Mock MongoClient and database
        mock_date = Mock()
        mock_date.date.return_value = 'test_date'
        mock_collection = Mock()
        mock_collection.find_one.return_value = {'created_at': mock_date}
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_database.list_collection_names.return_value = ['raw']
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        mock_client.list_database_names.return_value = ['test_dataset']
        with patch('figures.figures.MongoClient', return_value=mock_client):
            date_range = self.mongo_plot.get_date_range("test_dataset")
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
        mock_database.list_collection_names.return_value = ['raw']
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        mock_client.list_database_names.return_value = ['test_dataset']
        with patch('figures.figures.MongoClient', return_value=mock_client):
            hashtag_freqs = self.mongo_plot.get_hashtag_freqs("test_dataset")
            self.assertEqual(hashtag_freqs, [(x['_id'], x['count']) for x in mock_collection.aggregate.return_value])

    def test_get_hashtag_freqs_2(self):
        expected = [('test_hashtag', 3), ('test_hashtag2', 2)]
        actual = self.mongo_plot.get_hashtag_freqs("test_dataset")
        self.assertEqual(expected, actual)

    def test_get_hashtag_freqs_3(self):
        self.assertRaises(RuntimeError, self.mongo_plot.get_hashtag_freqs, "test_not_exists")

    def test_get_user_id(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_database.list_collection_names.return_value = ['raw']
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        mock_client.list_database_names.return_value = ['test_dataset']
        # Mock find_one
        mock_collection.find_one.return_value = {
            'author': {'username': 'test_username', 'id': 'test_id'}
        }

        with patch('figures.figures.MongoClient', return_value=mock_client):
            user_id = self.mongo_plot.get_user_id("test_dataset", "test_username")
            self.assertEqual(user_id, 'test_id')

    def test_get_user_id_2(self):
        expected = 0
        actual = self.mongo_plot.get_user_id("test_dataset", "TEST_USER_0")
        self.assertEqual(expected, actual)

    def test_available_datasets(self):
        # Mock MongoClient
        with patch('figures.figures.MongoClient') as mock_client:
            mock_client.return_value.list_database_names.return_value = ['test_dataset']
            datasets = self.mongo_plot.available_datasets
            self.assertEqual(datasets, ['test_dataset'])

    def test__get_hashtag_freqs(self):
        expected = [('test_hashtag', 3), ('test_hashtag2', 2)]
        self.mongo_plot.max_hashtags = None
        actual = self.mongo_plot._get_hashtag_freqs("test_dataset")
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
