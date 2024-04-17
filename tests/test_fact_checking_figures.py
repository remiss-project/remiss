import unittest

from pymongo import MongoClient

from figures.fact_checking import FactCheckingPlotFactory

DATA_DIR = './../fact_checking_data'
class TestTimeSeriesFactory(unittest.TestCase):
    def setUp(self):
        self.plot_factory = FactCheckingPlotFactory(data_dir=DATA_DIR)
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('fact_checking')
        self.database = self.client.get_database('fact_checking')
        self.collection = self.database.get_collection('test_dataset')
        test_data = [{"id": 47, 'tweet_id': '100485425'},
                     {'id': 67, 'tweet_id': '100485426'}]
        self.collection.insert_many(test_data)

    def test_fact_checking_plot(self):
        plot = self.plot_factory.plot_fact_checking('test_dataset', '100485425')
        with open(DATA_DIR + '/test_dataset/47.htm', 'r') as f:
            data = f.read()
        assert plot == data

    def test_get_fact_checking_data_id(self):
        data_id = self.plot_factory.get_fact_checking_data_id('test_dataset', '100485425')
        assert data_id == 47

    def test_load_data_for_tweet(self):
        data = self.plot_factory.load_data_for_tweet('test_dataset', 47)
        # check data is html
        assert "</html>" in data