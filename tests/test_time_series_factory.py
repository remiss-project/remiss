import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
from plotly.graph_objs import Figure
from pymongo import MongoClient

from figures import TimeSeriesFactory


class TestTimeSeriesFactory(unittest.TestCase):
    def setUp(self):
        self.tweet_user_plot = TimeSeriesFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_remiss')
        self.database = self.client.get_database('test_remiss')
        self.collection = self.database.get_collection('test_collection')
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

    def test_plot_tweet_series(self):
        # Mock _get_counts
        self.tweet_user_plot._get_count_area_plot = Mock(return_value='mocked_plot_data')

        collection = 'test_collection'
        hashtag = 'test_hashtag'
        start_time = '2023-01-01'
        end_time = '2023-01-31'
        unit = 'day'
        bin_size = 1
        expected_pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]

        result = self.tweet_user_plot.plot_tweet_series(collection, hashtag, start_time, end_time, unit, bin_size)

        self.assertEqual(result, 'mocked_plot_data')
        self.tweet_user_plot._get_count_area_plot.assert_called_with(
            expected_pipeline, collection, hashtag, start_time, end_time
        )

    def test_plot_tweet_series_2(self):
        from numpy import nan
        actual = self.tweet_user_plot.plot_tweet_series('test_collection', None, None, None)
        self.assertEqual([datetime(2019, 1, 1, 0, 0),
                          datetime(2019, 1, 2, 0, 0),
                          datetime(2019, 1, 3, 0, 0)], list(actual['data'][0]['x']))
        actual = pd.DataFrame([actual['data'][i]['y'] for i in range(len(actual['data']))])
        expected = pd.DataFrame({0: {0: nan, 1: nan, 2: 1.0, 3: nan}, 1: {0: 1.0, 1: nan, 2: nan, 3: nan},
                                 2: {0: nan, 1: nan, 2: nan, 3: 1.0}})
        pd.testing.assert_frame_equal(actual, expected)

    def test_plot_user_series(self):
        # Mock _get_counts
        self.tweet_user_plot._get_count_area_plot = Mock(return_value='mocked_plot_data')

        collection = 'test_collection'
        hashtag = 'test_hashtag'
        start_time = '2023-01-01'
        end_time = '2023-01-31'
        unit = 'day'
        bin_size = 1
        expected_pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "users": {'$addToSet': "$author.username"}
            }
            },
            {'$project': {'count': {'$size': '$users'}}},
            {'$sort': {'_id': 1}}
        ]
        result = self.tweet_user_plot.plot_user_series(collection, hashtag, start_time, end_time, unit, bin_size)

        self.assertEqual(result, 'mocked_plot_data')
        self.tweet_user_plot._get_count_area_plot.assert_called_with(
            expected_pipeline, collection, hashtag, start_time, end_time
        )

    def test_plot_user_series_2(self):
        from numpy import nan
        actual = self.tweet_user_plot.plot_user_series('test_collection', None, None, None)
        self.assertEqual([datetime(2019, 1, 1, 0, 0),
                          datetime(2019, 1, 2, 0, 0),
                          datetime(2019, 1, 3, 0, 0)], list(actual['data'][0]['x']))
        actual = pd.DataFrame([actual['data'][i]['y'] for i in range(len(actual['data']))])
        expected = pd.DataFrame({0: {0: nan, 1: nan, 2: 1.0, 3: nan}, 1: {0: 1.0, 1: nan, 2: nan, 3: nan},
                                 2: {0: nan, 1: nan, 2: nan, 3: 1.0}})
        pd.testing.assert_frame_equal(actual, expected)

    @patch('figures.time_series.MongoClient')
    def test__get_counts(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        # Mock aggregate_pandas_all
        mock_collection.aggregate_pandas_all.return_value = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                                                                  datetime(2023, 1, 2),
                                                                                  datetime(2023, 1, 3)],
                                                                          'count': [1, 2, 3]})

        # Mock _add_filters
        self.tweet_user_plot._add_filters = Mock(return_value='mocked_pipeline')

        collection = 'test_collection'
        hashtag = 'test_hashtag'
        start_time = '2023-01-01'
        end_time = '2023-01-31'
        unit = 'day'
        bin_size = 1

        expected_pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]
        result = self.tweet_user_plot._get_count_area_plot(
            expected_pipeline, collection, hashtag, start_time, end_time
        )

        self.assertIsInstance(result, Figure)

        mock_mongo_client.assert_called_with(self.tweet_user_plot.host, self.tweet_user_plot.port)
        mock_database.get_collection.assert_called_with(collection)

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

        result = self.tweet_user_plot._add_filters([], hashtag, start_time, end_time, user_type)

        self.assertEqual(result, expected_pipeline)

    def test__add_filters_normal(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': None}}]
        result = self.tweet_user_plot._add_filters([], hashtags=None, start_time=None, end_time=None,
                                                   user_type='normal')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_suspect(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                         'author.remiss_metadata.party': None}}]
        result = self.tweet_user_plot._add_filters([], hashtags=None, start_time=None, end_time=None,
                                                   user_type='suspect')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_politician(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': {'$ne': None}}}]
        result = self.tweet_user_plot._add_filters([], hashtags=None, start_time=None, end_time=None,
                                                   user_type='politician')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_suspect_politician(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                         'author.remiss_metadata.party': {'$ne': None}}}]
        result = self.tweet_user_plot._add_filters([], hashtags=None, start_time=None, end_time=None,
                                                   user_type='suspect_politician')
        self.assertEqual(result, expected_pipeline)

    def test__get_count_data(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []

        collection = Mock()
        test_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          datetime(2023, 1, 3)],
                                  'count': [1, 2, 3]})
        collection.aggregate_pandas_all.return_value = test_data
        actual = self.tweet_user_plot._get_count_data(pipeline, hashtag, start_time, end_time, collection)
        test_data = test_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([test_data] * 4, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test__get_count_data_empty(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []

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
        actual = self.tweet_user_plot._get_count_data(pipeline, hashtag, start_time, end_time, collection)
        good_data = good_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        bad_data = bad_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([good_data] * 2 + [bad_data] * 2, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        pd.testing.assert_frame_equal(actual, expected)

    def test__get_count_data_index_unmatched(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []

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
        actual = self.tweet_user_plot._get_count_data(pipeline, hashtag, start_time, end_time, collection)
        good_data = good_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        bad_data = bad_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([good_data] * 2 + [bad_data] * 2, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        pd.testing.assert_frame_equal(actual, expected)
