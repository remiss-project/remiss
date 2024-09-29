import unittest
from unittest.mock import Mock, patch

import pandas as pd
from pymongo import MongoClient

from figures.control import ControlPlotFactory
from figures.figures import MongoPlotFactory


class TestMongoPlotFactory(unittest.TestCase):
    def setUp(self):
        self.control_plot_factory = ControlPlotFactory()
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


    def test_get_hashtag_freqs(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        test_data = pd.DataFrame([{'hashtag': 'test_hashtag', 'count': 3},
                                  {'hashtag': 'test_hashtag2', 'count': 2}])

        mock_collection.aggregate_pandas_all.return_value = test_data
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_database.list_collection_names.return_value = ['raw']
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        mock_client.list_database_names.return_value = ['test_dataset']
        with patch('figures.control.MongoClient', return_value=mock_client):
            hashtag_freqs = self.control_plot_factory._compute_hashtag_freqs("test_dataset")
            pd.testing.assert_frame_equal(test_data, hashtag_freqs)

    def test_persistence(self):
        expected_hashtag_freqs = self.control_plot_factory._compute_hashtag_freqs("test_dataset_2")
        self.control_plot_factory.persist(['test_dataset_2'])
        actual_hashtag_freqs = self.control_plot_factory._load_hashtag_freqs('test_dataset_2')
        pd.testing.assert_frame_equal(expected_hashtag_freqs, actual_hashtag_freqs)


if __name__ == '__main__':
    unittest.main()
