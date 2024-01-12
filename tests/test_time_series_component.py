import unittest
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

from dash import Dash
from dash.dcc import DatePickerRange, Graph
from dash_holoniq_wordcloud import DashWordcloud

from components.dashboard import RemissState
from components.time_series import TimeSeriesComponent


class TimeSeriesComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = Mock()
        self.plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                         datetime(2023, 12, 31))
        self.plot_factory.plot_tweet_series.return_value = 'plot_tweet_series'
        self.plot_factory.plot_user_series.return_value = 'plot_user_series'
        self.plot_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                            ('hashtag3', 3), ('hashtag4', 2),
                                                            ('hashtag5', 1), ('hashtag6', 1), ]
        self.state = RemissState(name='state')
        self.component = TimeSeriesComponent(self.plot_factory, self.state, name='timeseries')

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a DatePickerRange
        # - a Wordcloud
        # - a TimeSeriesComponent
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Graph, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, DatePickerRange):
                found_components.append(component)
            if isinstance(component, DashWordcloud):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn('fig-tweet', component_ids)
        self.assertIn('fig-users', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def test_update_callback(self):
        app = Dash()
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'TimeSeriesComponent.update' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'current-dataset-state', 'property': 'data'},
                                              {'id': 'current-hashtags-state', 'property': 'data'},
                                              {'id': 'current-start-date-state', 'property': 'data'},
                                              {'id': 'current-end-date-state', 'property': 'data'}])
        actual_output = [{'component_id': o.component_id, 'property': o.component_property} for o in
                         callback['output']]
        self.assertEqual(actual_output, [{'component_id': 'fig-tweet-timeseries', 'property': 'figure'},
                                         {'component_id': 'fig-users-timeseries', 'property': 'figure'}])


if __name__ == '__main__':
    unittest.main()
