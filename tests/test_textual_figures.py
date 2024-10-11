import unittest
import uuid
from datetime import datetime
from unittest import TestCase

from plotly.graph_objs import Figure
from pymongo import MongoClient

from figures.textual import TextualFactory


class TestTextualFigures(TestCase):
    def setUp(self):
        self.plot_factory = TextualFactory()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_plot_emotion_per_hour(self):
        fig = self.plot_factory.plot_emotion_per_hour(self.test_dataset, None, None)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 13

    def test_plot_emotion_per_hour_start_date(self):
        start_date = datetime.fromisoformat("2015-10-12 15:05:49")
        fig = self.plot_factory.plot_emotion_per_hour(self.test_dataset, start_date, None)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 13

    def test_plot_emotion_per_hour_end_date(self):
        end_date = datetime.fromisoformat("2010-10-12 15:05:49")
        with self.assertRaises(ValueError):
            self.plot_factory.plot_emotion_per_hour(self.test_dataset, None, end_date)

    def test_plot_emotion_per_hour_start_date_none(self):
        start_date = datetime.fromisoformat("2024-10-12 15:05:49")
        with self.assertRaises(ValueError):
            self.plot_factory.plot_emotion_per_hour(self.test_dataset, start_date, None)

    def test_plot_emotion_per_hour_end_date_none(self):
        end_date = datetime.fromisoformat("2015-10-12 15:05:49")
        fig = self.plot_factory.plot_emotion_per_hour(self.test_dataset, None, end_date)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 13

    def test_plot_average_emotion(self):
        fig = self.plot_factory.plot_average_emotion(self.test_dataset)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 11

    @unittest.skip("Won't be used")
    def test_plot_top_profiles(self):
        fig = self.plot_factory.plot_top_profiles(self.test_dataset, None, None)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2

    @unittest.skip("Won't be used")
    def test_plot_top_hashtags(self):
        fig = self.plot_factory.plot_top_hashtags(self.test_dataset, None, None)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2

    @unittest.skip("Won't be used")
    def test_plot_topic_ranking(self):
        fig = self.plot_factory.plot_topic_ranking(self.test_dataset, None, None)
        # fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2
