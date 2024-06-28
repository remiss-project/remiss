from unittest import TestCase

from plotly.graph_objs import Figure

from figures.textual import TextualFactory
from tests.conftest import populate_test_database


class TestTextualFigures(TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     populate_test_database('test_dataset')

    def test_plot_emotion_per_hour(self):
        plot_factory = TextualFactory()
        fig = plot_factory.plot_emotion_per_hour('test_dataset', None, None)
        fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 13

    def test_plot_average_emotion(self):
        plot_factory = TextualFactory()
        fig = plot_factory.plot_average_emotion('test_dataset', None, None)
        fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 11

    def test_plot_top_profiles(self):
        plot_factory = TextualFactory()
        fig = plot_factory.plot_top_profiles('test_dataset', None, None)
        fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2

    def test_plot_top_hashtags(self):
        plot_factory = TextualFactory()

        fig = plot_factory.plot_top_hashtags('test_dataset', None, None)
        fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2

    def test_plot_topic_ranking(self):
        plot_factory = TextualFactory()
        fig = plot_factory.plot_topic_ranking('test_dataset', None, None)
        fig.show()
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2
