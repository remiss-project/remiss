import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
from pymongo import MongoClient

from propagation.histogram import Histogram


class TestHistogram(unittest.TestCase):
    def setUp(self):
        self.histogram = Histogram()
        self.client = MongoClient('localhost', 27017)
        self.dataset = 'test_dataset'
        self.client.drop_database(self.dataset)
        self.database = self.client.get_database(self.dataset)
        self.collection = self.database.get_collection('raw')
        test_data = [{"id": 0, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                      "author": {"username": "TEST_USER_0", "id": 0,
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": 1, "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      "author": {"username": "TEST_USER_1", "id": 1,
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "quoted"}]},
                     {"id": 2, "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      "author": {"username": "TEST_USER_2", "id": 2,
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

    def test__add_filters(self):
        hashtag = ['test_hashtag']
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        user_type = 'normal'
        expected_pipeline = [{'$match': {'created_at': {'$lte': end_time}}},
                             {'$match': {'created_at': {'$gte': start_time}}},
                             {'$match': {'entities.hashtags.tag': hashtag[0]}},
                             {'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': None}}]

        result = self.histogram._add_filters([], hashtag, start_time, end_time, user_type)

        self.assertEqual(result, expected_pipeline)

    def test__add_filters_normal(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': None}}]
        result = self.histogram._add_filters([], hashtags=None, start_time=None, end_time=None,
                                             user_type='normal')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_suspect(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                         'author.remiss_metadata.party': None}}]
        result = self.histogram._add_filters([], hashtags=None, start_time=None, end_time=None,
                                             user_type='suspect')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_politician(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': {'$ne': None}}}]
        result = self.histogram._add_filters([], hashtags=None, start_time=None, end_time=None,
                                             user_type='politician')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_suspect_politician(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                         'author.remiss_metadata.party': {'$ne': None}}}]
        result = self.histogram._add_filters([], hashtags=None, start_time=None, end_time=None,
                                             user_type='suspect_politician')
        self.assertEqual(result, expected_pipeline)

    @patch('propagation.histogram.MongoClient')
    def test__get_count_data(self, mock_mongo_client):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        dataset = 'test_dataset'
        hashtags = ['test_hashtag']
        pipeline = []

        collection = Mock()
        test_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          datetime(2023, 1, 3)],
                                  'count': [1, 2, 3]})

        collection.aggregate_pandas_all.return_value = test_data
        mock_mongo_client.return_value.get_database.return_value.get_collection.return_value = collection
        mock_mongo_client.return_value.list_database_names.return_value = [dataset]
        mock_mongo_client.return_value.get_database.return_value.list_collection_names.return_value = ['raw']
        actual = self.histogram._get_count_data(dataset, pipeline, hashtags, start_time, end_time)
        test_data = test_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([test_data] * 4, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        self.assertEqual(actual.to_dict(), expected.to_dict())

    @patch('propagation.histogram.MongoClient')
    def test__get_count_data_empty(self, mock_mongo_client):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []
        dataset = 'test_dataset'
        hashtags = ['test_hashtag']

        collection = Mock()
        good_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          datetime(2023, 1, 3)],
                                  'count': [1, 2, 3]})
        bad_data = pd.DataFrame({'_id': [],
                                 'count': []})

        class TestData:
            def __init__(self):
                self.i = 0

            def get_data(self, pipeline, schema):
                if self.i < 2:
                    data = good_data
                else:
                    data = bad_data
                self.i += 1
                return data

        test_data = TestData()
        collection.aggregate_pandas_all = test_data.get_data
        mock_mongo_client.return_value.get_database.return_value.get_collection.return_value = collection
        mock_mongo_client.return_value.list_database_names.return_value = [dataset]
        mock_mongo_client.return_value.get_database.return_value.list_collection_names.return_value = ['raw']
        actual = self.histogram._get_count_data(dataset, pipeline, hashtags, start_time, end_time)
        good_data = good_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        bad_data = bad_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([good_data] * 2 + [bad_data] * 2, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        expected = expected.fillna(0)
        pd.testing.assert_frame_equal(actual, expected)

    @patch('propagation.histogram.MongoClient')
    def test__get_count_data_index_unmatched(self, mock_mongo_client):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []
        dataset = 'test_dataset'
        hashtags = ['test_hashtag']

        collection = Mock()
        good_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          ],
                                  'count': [1, 2]})
        bad_data = pd.DataFrame({'_id': [
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)],
            'count': [2, 3]})

        class TestData:
            def __init__(self):
                self.i = 0

            def get_data(self, pipeline, schema):
                if self.i < 2:
                    data = good_data
                else:
                    data = bad_data
                self.i += 1
                return data

        test_data = TestData()
        collection.aggregate_pandas_all = test_data.get_data
        mock_mongo_client.return_value.get_database.return_value.get_collection.return_value = collection
        mock_mongo_client.return_value.list_database_names.return_value = [dataset]
        mock_mongo_client.return_value.get_database.return_value.list_collection_names.return_value = ['raw']
        actual = self.histogram._get_count_data(dataset, pipeline, hashtags, start_time, end_time)
        good_data = good_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        bad_data = bad_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([good_data] * 2 + [bad_data] * 2, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        expected = expected.fillna(0)
        pd.testing.assert_frame_equal(actual, expected)

    def test_persistence(self):
        self.histogram.persist(['test_dataset_2'])
        actual_tweet = self.histogram.load_histogram('test_dataset_2', 'tweet')
        actual_user = self.histogram.load_histogram('test_dataset_2', 'user')

        expected_tweet = self.histogram.compute_tweet_histogram('test_dataset_2', None, None, None)
        expected_user = self.histogram.compute_user_histogram('test_dataset_2', None, None, None)

        pd.testing.assert_frame_equal(actual_tweet, expected_tweet, check_dtype=False)
        pd.testing.assert_frame_equal(actual_user, expected_user, check_dtype=False)
