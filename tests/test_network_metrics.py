import time
import unittest

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pandas import Timestamp
from pymongo import MongoClient
from scipy.stats import rankdata

from propagation import NetworkMetrics
from tests.conftest import create_test_data


class NetworkMetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.network_metrics = NetworkMetrics()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_legitimacy(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data(day_range=day_range)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        self.network_metrics.unit = 'day'
        self.network_metrics.bin_size = 20 + day_range
        actual = self.network_metrics.compute_legitimacy(self.tmp_dataset)
        expected = []
        for t in test_data:
            for referenced_tweet in t['referenced_tweets']:
                expected.append({'author_id': referenced_tweet['author']['id'],
                                 'legitimacy': 1})

        expected = pd.DataFrame(expected)

        expected = expected.groupby(['author_id']).count().sort_values('legitimacy', ascending=False)['legitimacy']

        pd.testing.assert_series_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False)

    def test_reputation(self):
        # compute reputation as the amount of referenced tweets attained by each user
        data_size = 100
        day_range = 10
        max_num_references = 20
        test_data = create_test_data(day_range=day_range, data_size=data_size, max_num_references=max_num_references)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        self.network_metrics.host = 'localhost'
        self.network_metrics.port = 27017
        actual = self.network_metrics.compute_reputation(self.tmp_dataset)
        expected = []
        for t in test_data:
            for referenced_tweet in t['referenced_tweets']:
                expected.append({'author_id': referenced_tweet['author']['id'],
                                 'date': t['created_at'].date(),
                                 'legitimacy': 1})

        expected = pd.DataFrame(expected)

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.cumsum(axis=1)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)

    def test__get_legitimacy_over_time_1(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data(day_range=day_range)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        self.network_metrics.unit = 'day'
        self.network_metrics.bin_size = 20 + day_range
        actual = self.network_metrics._get_legitimacy_over_time(self.tmp_dataset)
        expected = []
        for t in test_data:
            for referenced_tweet in t['referenced_tweets']:
                expected.append({'author_id': referenced_tweet['author']['id'],
                                 'date': t['created_at'].date(),
                                 'legitimacy': 1})

        expected = pd.DataFrame(expected)

        expected = expected.groupby(['author_id']).count()

        expected = expected['legitimacy'].sort_index()
        actual = actual.iloc[:, 0].sort_index()
        pd.testing.assert_series_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False,
                                       check_names=False)

    def test__get_legitimacy_over_time_2(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data(day_range=day_range)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.network_metrics._get_legitimacy_over_time(self.tmp_dataset)
        expected = []
        for t in test_data:
            for referenced_tweet in t['referenced_tweets']:
                expected.append({'author_id': referenced_tweet['author']['id'],
                                 'date': t['created_at'].date(),
                                 'legitimacy': 1})

        expected = pd.DataFrame(expected)

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)

    def test_status(self):
        # compute status as the amount of referenced tweets attained by each user
        day_range = 10
        test_data = create_test_data(day_range=day_range)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.network_metrics.compute_status(self.tmp_dataset)
        expected = []
        for t in test_data:
            for referenced_tweet in t['referenced_tweets']:
                expected.append({'author_id': referenced_tweet['author']['id'],
                                 'date': t['created_at'].date(),
                                 'legitimacy': 1})

        expected = pd.DataFrame(expected)

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.fillna(0).cumsum(axis=1)
        expected = expected.apply(lambda x: rankdata(x, method='min', nan_policy='omit'))
        expected = expected - 1
        actual = actual.sort_index()
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False,
                                      check_column_type=False)

    def test_legitimacy_full(self):
        actual = self.network_metrics.compute_legitimacy(self.test_dataset)
        actual = actual.to_list()[:5]
        expected = [797.0, 733.0, 366.0, 265.0, 161.0]
        self.assertEqual(actual, expected)

    def test_legitimacy_no_nans(self):
        actual = self.network_metrics.compute_legitimacy(self.test_dataset)
        self.assertFalse(actual.isnull().values.any())

    def test_no_missing_people(self):
        test_author_id = '280227352'
        actual = self.network_metrics.compute_legitimacy(self.test_dataset)

        assert test_author_id in actual.index
        print(actual.loc[test_author_id])

    def test_reputation_full(self):
        actual = self.network_metrics.compute_reputation(self.test_dataset)
        self.assertIsInstance(actual.columns, pd.DatetimeIndex)
        self.assertEquals(str(actual.index.dtype), 'object')
        self.assertGreater(actual.shape[1], 0)
        self.assertGreater(actual.shape[0], 0)

    def test__get_legitimacy_over_time_1_full(self):
        actual = self.network_metrics._get_legitimacy_over_time(self.test_dataset)
        self.assertIsInstance(actual.columns, pd.DatetimeIndex)
        self.assertEquals(str(actual.index.dtype), 'object')
        self.assertGreater(actual.shape[1], 0)
        self.assertGreater(actual.shape[0], 0)

    def test_status_full(self):
        actual = self.network_metrics.compute_status(self.test_dataset)
        self.assertIsInstance(actual.columns, pd.DatetimeIndex)
        self.assertEquals(str(actual.index.dtype), 'object')
        self.assertGreater(actual.shape[1], 0)
        self.assertGreater(actual.shape[0], 0)

    def test_persistence_and_loading_full(self):
        # Test the persistence and loading of the graph
        start_time = Timestamp.now()
        self.network_metrics.persist([self.test_dataset])
        end_time = Timestamp.now()
        print(f'persisted in {end_time - start_time}')

        start_time = time.time()
        expected_legitimacy = self.network_metrics.compute_legitimacy(self.test_dataset)
        expected_reputation = self.network_metrics.compute_reputation(self.test_dataset)
        expected_status = self.network_metrics.compute_status(self.test_dataset)

        end_time = time.time()
        print(f'computed in {end_time - start_time} seconds')

        actual_legitimacy = self.network_metrics.get_legitimacy(self.test_dataset)
        actual_reputation = self.network_metrics.get_reputation(self.test_dataset)
        actual_status = self.network_metrics.get_status(self.test_dataset)

        pd.testing.assert_series_equal(expected_legitimacy, actual_legitimacy, check_dtype=False, check_like=True,
                                       check_index_type=False)
        pd.testing.assert_frame_equal(expected_reputation, actual_reputation, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)
        pd.testing.assert_frame_equal(expected_status, actual_status, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)

    def test_plot_histograms(self):
        # Test the plotting of histograms
        legitimacy = self.network_metrics.compute_legitimacy(self.test_dataset)
        mean_reputation = self.network_metrics.compute_reputation(self.test_dataset).mean(axis=1)
        mean_status = self.network_metrics.compute_status(self.test_dataset).mean(axis=1)

        fig = px.histogram(x=legitimacy, title='Legitimacy')
        # fig.show()
        fig = px.histogram(x=mean_reputation, title='Reputation')
        # fig.show()
        fig = px.histogram(x=mean_status, title='Status')
        # fig.show()

    def test_plot_kde(self):
        # Test the plotting of histograms
        legitimacy = self.network_metrics.compute_legitimacy(self.test_dataset)
        mean_reputation = self.network_metrics.compute_reputation(self.test_dataset).mean(axis=1)
        mean_status = self.network_metrics.compute_status(self.test_dataset).mean(axis=1)

        fig = ff.create_distplot([legitimacy], group_labels=['Legitimacy'])
        # fig.show()
        fig = ff.create_distplot([mean_reputation], group_labels=['Reputation'])
        # fig.show()
        fig = ff.create_distplot([mean_status], group_labels=['Status'])
        # fig.show()

    def test_get_level(self):
        legitimacy = pd.Series([np.nan, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        reputation = pd.Series([np.nan, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        status = pd.Series([np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        legitimacy_level = self.network_metrics.get_level(legitimacy)
        reputation_level = self.network_metrics.get_level(reputation)
        status_level = self.network_metrics.get_level(status)

        expected = ['Null', 'Null', 'Low', 'Low', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High', 'High']
        self.assertEqual(legitimacy_level.tolist(), expected)
        self.assertEqual(reputation_level.tolist(), expected)
        self.assertEqual(status_level.tolist(), expected)

    def test_get_level_2(self):
        legitimacy = pd.Series([np.nan, 0, 0.1, 0.1, 0.11, 0.12, 0.13, 0.13, 0.14, 0.15, 0.9, 1.0])
        reputation = pd.Series([np.nan, 0, 0.1, 0.1, 0.11, 0.12, 0.13, 0.13, 0.14, 0.15, 0.9, 1.0])
        status = pd.Series([np.nan, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        legitimacy_level = self.network_metrics.get_level(legitimacy)
        reputation_level = self.network_metrics.get_level(reputation)
        status_level = self.network_metrics.get_level(status)

        expected = ['Null', 'Null', 'Low', 'Low', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High', 'High']
        self.assertEqual(legitimacy_level.tolist(), expected)
        self.assertEqual(reputation_level.tolist(), expected)
        self.assertEqual(status_level.tolist(), expected)

    def test_network_metrics_get_levels(self):
        legitimacy = self.network_metrics.compute_legitimacy(self.test_dataset)
        reputation = self.network_metrics.compute_reputation(self.test_dataset).mean(axis=1)
        status = self.network_metrics.compute_status(self.test_dataset).mean(axis=1)

        legitimacy_level = self.network_metrics.get_level(legitimacy)
        reputation_level = self.network_metrics.get_level(reputation)
        status_level = self.network_metrics.get_level(status)

        self.assertEqual(legitimacy_level.shape, legitimacy.shape)
        self.assertEqual(reputation_level.shape, reputation.shape)
        self.assertEqual(status_level.shape, status.shape)

        self.assertEqual(legitimacy_level.cat.categories.tolist(), ['Null', 'Low', 'High'])
        self.assertEqual(reputation_level.cat.categories.tolist(), ['Null', 'Low', 'Medium', 'High'])
        self.assertEqual(status_level.cat.categories.tolist(), ['Null', 'Low', 'Medium',  'High'])

    def test_get_legitimacy_for_author(self):
        author_id = '280227352'
        actual = self.network_metrics.load_legitimacy_for_author(self.test_dataset, author_id)
        self.assertIsInstance(actual, (int, float))

    def test_get_reputation_for_author(self):
        author_id = '280227352'
        actual = self.network_metrics.load_reputation_for_author(self.test_dataset, author_id)
        self.assertIsInstance(actual, pd.Series)
        self.assertGreater(actual.shape[0], 0)

    def test_get_status_for_author(self):
        author_id = '280227352'
        actual = self.network_metrics.load_status_for_author(self.test_dataset, author_id)
        self.assertIsInstance(actual, pd.Series)
        self.assertGreater(actual.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
