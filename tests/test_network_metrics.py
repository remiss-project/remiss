import time
import unittest
import uuid

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pymongo import MongoClient

from propagation import NetworkMetrics
from tests.conftest import create_test_data


class NetworkMetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.network_metrics = NetworkMetrics()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)

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
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id'])['legitimacy'].sum()

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
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

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
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id'])['legitimacy'].sum()

        expected = expected.to_frame()
        expected.columns = actual.columns
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False)

    def test__get_legitimacy_over_time_2(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data(day_range=day_range)
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.network_metrics._get_legitimacy_over_time(self.tmp_dataset)
        expected = pd.DataFrame({
            'author_id': [t['author']['id'] for t in test_data],
            'date': [t['created_at'].date() for t in test_data],
            'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

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
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.cumsum(axis=1)
        expected = expected.apply(lambda x: x.argsort())
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False,
                                      check_column_type=False)

    def test_legitimacy_full(self):
        actual = self.network_metrics.compute_legitimacy(self.test_dataset)
        actual = actual.to_list()[:5]
        expected = [238, 233, 202, 195, 148]
        self.assertEqual(actual, expected)

    def test_reputation_full(self):
        actual = self.network_metrics.compute_reputation(self.test_dataset)
        self.assertEqual(actual.shape[1], 272)
        self.assertEqual(actual.shape[0], 2578)

    def test__get_legitimacy_over_time_1_full(self):
        actual = self.network_metrics._get_legitimacy_over_time(self.test_dataset)
        self.assertEqual(actual.shape[1], 272)
        self.assertEqual(actual.shape[0], 2578)

    def test_status_full(self):
        actual = self.network_metrics.compute_status(self.test_dataset)
        self.assertEqual(actual.shape[1], 272)
        self.assertEqual(actual.shape[0], 2578)

    def test_persistence_and_loading_full(self):
        # Test the persistence and loading of the graph
        self.network_metrics.persist([self.test_dataset])

        start_time = time.time()
        self.network_metrics.load_from_mongodb([self.test_dataset])
        end_time = time.time()
        print(f'loaded in {end_time - start_time} seconds')

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
        fig.show()
        fig = px.histogram(x=mean_reputation, title='Reputation')
        fig.show()
        fig = px.histogram(x=mean_status, title='Status')
        fig.show()

    def test_plot_kde(self):
        # Test the plotting of histograms
        legitimacy = self.network_metrics.compute_legitimacy(self.test_dataset)
        mean_reputation = self.network_metrics.compute_reputation(self.test_dataset).mean(axis=1)
        mean_status = self.network_metrics.compute_status(self.test_dataset).mean(axis=1)

        fig = ff.create_distplot([legitimacy], group_labels=['Legitimacy'])
        fig.show()
        fig = ff.create_distplot([mean_reputation], group_labels=['Reputation'])
        fig.show()
        fig = ff.create_distplot([mean_status], group_labels=['Status'])
        fig.show()


if __name__ == '__main__':
    unittest.main()
