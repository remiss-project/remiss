from datetime import datetime
from unittest import TestCase

from plotly.graph_objs import Figure

from figures.textual import TextualFactory


class TestTextualFigures(TestCase):
    def setUp(self):
        self.plot_factory = TextualFactory()
        self.test_dataset = 'test_dataset_2'

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

    def test_remote_emotion_per_hour(self):
        self.plot_factory.host = 'mongodb://srvinv02.esade.es'
        datasets = [
            'Openarms', 'MENA_Agressions', 'MENA_Ajudes', 'Barcelona_2019', 'Andalucia_2022',
            'Generales_2019',
            'Generalitat_2021',
        ]

        for dataset in datasets:
            print(f'Plotting {dataset}')
            fig = self.plot_factory.plot_emotion_per_hour(dataset, None, None)
            fig.update_layout(title=dataset)
            fig.show()
            assert isinstance(fig, Figure)
            assert len(fig.data) == 14
            for trace in fig.data:
                assert len(trace.x) >= 24
                assert len(trace.y) >= 24
