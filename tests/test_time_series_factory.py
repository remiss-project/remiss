import unittest
import uuid

from plotly.graph_objs import Figure
from pymongo import MongoClient

from figures import TimeSeriesFactory


class TestTimeSeriesFactory(unittest.TestCase):
    def setUp(self):
        self.tweet_user_plot = TimeSeriesFactory()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.test_tweet_id = '1167074391315890176'
        self.missing_tweet_id = '1077146799692021761'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_plot_tweet_series(self):
        start_time = '2003-01-01'
        end_time = '2023-01-31'

        result = self.tweet_user_plot.plot_tweet_series(self.test_dataset, None, start_time, end_time)

        self.assertIsInstance(result, Figure)

    def test_plot_user_series(self):
        start_time = '2003-01-01'
        end_time = '2023-01-31'

        result = self.tweet_user_plot.plot_user_series(self.test_dataset, None, start_time, end_time)

        self.assertIsInstance(result, Figure)
