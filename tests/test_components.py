import unittest
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

from dash import Dash
from dash.dcc import DatePickerRange, Graph
from dash_holoniq_wordcloud import DashWordcloud

from components import DashComponent, TweetUserTimeSeriesComponent


class TestDashComponent(TestCase):
    def test_name(self):
        class TestComponent(DashComponent):
            def __init__(self, name=None):
                super().__init__(name=name)

        component = TestComponent()
        self.assertEqual(len(component.name), 10)


class TweetUserTimeSeriesComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = Mock()
        self.plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                         datetime(2023, 12, 31))
        self.component = TweetUserTimeSeriesComponent(self.plot_factory)

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a DatePickerRange
        # - a Wordcloud
        # - a TweetUserTimeSeriesComponent
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
        found_components = [type(component) for component in found_components]
        self.assertIn(DatePickerRange, found_components)
        self.assertIn(DashWordcloud, found_components)
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
        component_ids = ['-'.join(component.id.split('-')[:-2]) for component in found_components]
        self.assertIn('date-picker', component_ids)
        self.assertIn('wordcloud', component_ids)
        self.assertIn('tweet', component_ids)
        self.assertIn('users', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-2:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def test_update_date_range_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the date range
        date_range_key = (f'..date-picker-{self.component.name}.min_date_allowed...'
                          f'date-picker-{self.component.name}.max_date_allowed...'
                          f'date-picker-{self.component.name}.start_date...'
                          f'date-picker-{self.component.name}.end_date..')
        callback = app.callback_map[date_range_key]
        self.assertEqual(callback['inputs'], [{'id': 'dropdown-dataset', 'property': 'value'}])
        expected_outputs = [f'date-picker-{self.component.name}.' + field for field in
                            ['min_date_allowed', 'max_date_allowed', 'start_date', 'end_date']]
        actual_outputs = [output.component_id + '.' + output.component_property for output in callback['output']]
        self.assertEqual(actual_outputs, expected_outputs)
        self.component.update_date_picker('dataset2')
        self.assertEqual(self.plot_factory.get_date_range.call_args[0][0], 'dataset2')

    def test_update_wordcloud_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the wordcloud
        wordcloud_key = f'wordcloud-{self.component.name}.list'
        callback = app.callback_map[wordcloud_key]
        self.assertEqual(callback['inputs'], [{'id': 'dropdown-dataset', 'property': 'value'}])
        self.assertEqual(callback['output'].component_id, f'wordcloud-{self.component.name}')
        self.assertEqual(callback['output'].component_property, 'list')
        self.component.update_wordcloud('dataset2')
        self.assertEqual(self.plot_factory.get_hashtag_freqs.call_args[0][0], 'dataset2')

    def test_update_plots_callback(self):
        app = Dash()
        self.component.callbacks(app)

        # Simulate the update function for the plots
        plots_key = (f'..dropdown-dataset.value...'
                     f'date-picker-{self.component.name}.start_date...'
                     f'date-picker-{self.component.name}.end_date...'
                     f'wordcloud-{self.component.name}.click..')
        callback = app.callback_map[plots_key]
        self.assertEqual(callback['inputs'], [{'id': 'dropdown-dataset', 'property': 'value'},
                                                  {'id': f'date-picker-{self.component.name}', 'property': 'start_date'},
                                                  {'id': f'date-picker-{self.component.name}', 'property': 'end_date'},
                                                  {'id': f'wordcloud-{self.component.name}', 'property': 'click'}])
        expected_outputs = [f'fig-tweet-{self.component.name}.figure',
                            f'fig-users-{self.component.name}.figure']
        actual_outputs = [output.component_id + '.' + output.component_property for output in callback['output']]
        self.assertEqual(actual_outputs, expected_outputs)



if __name__ == '__main__':
    unittest.main()
