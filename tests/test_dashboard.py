import re
import unittest
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd
from dash import Dash
from dash.dcc import DatePickerRange, Graph, Dropdown, Slider
from dash_holoniq_wordcloud import DashWordcloud

from components import RemissDashboard


class RemissDashboardTest(TestCase):
    def setUp(self):
        self.tweet_user_plot_factory = Mock()
        self.tweet_user_plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.tweet_user_plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                    datetime(2023, 12, 31))
        self.tweet_user_plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.tweet_user_plot_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                                       ('hashtag3', 3), ('hashtag4', 2),
                                                                       ('hashtag5', 1), ('hashtag6', 1), ]
        self.propagation_factory = Mock()
        self.propagation_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.propagation_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                datetime(2023, 12, 31))
        self.propagation_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.tweet_table_factory = Mock()
        self.tweet_table_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.tweet_table_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                datetime(2023, 12, 31))
        self.tweet_table_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.tweet_table_factory.retweeted_table_columns = ['id', 'text', 'user']
        self.tweet_table_factory.user_table_columns = ['id', 'name', 'screen_name']
        self.tweet_table_factory.top_table_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party']

        self.textual_factory = Mock()
        self.textual_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.textual_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                            datetime(2023, 12, 31))
        self.textual_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.textual_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                               ('hashtag3', 3), ('hashtag4', 2),
                                                               ('hashtag5', 1), ('hashtag6', 1), ]

        self.multimodal_factory = Mock()
        self.multimodal_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.multimodal_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                               datetime(2023, 12, 31))
        self.multimodal_factory.get_users.return_value = ['user1', 'user2', 'user3']

        self.profile_factory = Mock()
        self.profile_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.profile_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                            datetime(2023, 12, 31))
        self.profile_factory.get_users.return_value = ['user1', 'user2', 'user3']

        self.control_plot_factory = Mock()
        self.control_plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.control_plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                datetime(2023, 12, 31)
                                                                )
        self.control_plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.control_plot_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                                   ('hashtag3', 3), ('hashtag4', 2),
                                                                   ('hashtag5', 1), ('hashtag6', 1), ]
        self.profiling_factory = Mock()
        self.profiling_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.profiling_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                              datetime(2023, 12, 31))
        self.profiling_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.profiling_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                                 ('hashtag3', 3), ('hashtag4', 2),
                                                                 ('hashtag5', 1), ('hashtag6', 1), ]


        self.component = RemissDashboard(
            self.control_plot_factory,
            self.tweet_user_plot_factory,
            self.tweet_table_factory,
            self.propagation_factory,
            self.textual_factory,
            self.profiling_factory,
            self.multimodal_factory,
            name='remiss-dashboard',
        )

        app = Dash(prevent_initial_callbacks="initial_duplicate",
                   )
        self.component.callbacks(app)

        callbacks = []
        for callback in app.callback_map.values():
            callbacks.append({'input': self.get_ids(callback['inputs']),
                              'output': self.get_ids(callback['output'])})

        callbacks = pd.DataFrame(callbacks)
        self.callbacks = {}
        for inputs, outputs in zip(callbacks['input'], callbacks['output']):
            for input_id in inputs:
                if input_id not in self.callbacks:
                    self.callbacks[input_id] = []
                self.callbacks[input_id].extend(outputs)

    def test_layout(self):
        layout = self.component.layout()

        def find_components(component, found_components):
            if hasattr(component, 'children') and component.children is not None:
                if len(component.children) == 0:
                    find_components(component.children, found_components)
                else:
                    for child in component.children:
                        find_components(child, found_components)
            if isinstance(component, Dropdown):
                found_components.append(component)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)
            if isinstance(component, DatePickerRange):
                found_components.append(component)
            if isinstance(component, DashWordcloud):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]

        self.assertIn(Dropdown, found_components)
        self.assertIn(Graph, found_components)
        self.assertIn(DashWordcloud, found_components)
        self.assertIn(DatePickerRange, found_components)
        self.assertIn(Slider, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # Pick all ids from the layout string
        ids = pd.Series(re.findall(r"id='(.*?)'", str(layout)))
        self.assertEqual(len(ids), 65)
        # Check for state
        assert ids.str.contains('state').any()
        # Check for upload
        assert ids.str.contains('upload').any()
        # Check for dataset dropdown
        assert ids.str.contains('dataset-dropdown').any()
        # Check for date range
        assert ids.str.contains('date-picker').any()
        # Check for wordcloud
        assert ids.str.contains('wordcloud').any()
        # Check for filter display
        assert ids.str.contains('display-filter').any()
        # Check for egonet
        assert ids.str.contains('egonet').any()
        # Check for time series
        assert ids.str.contains('time-series').any()
        # Check for profiling
        assert ids.str.contains('profiling').any()
        # Check for table
        assert ids.str.contains('tweet-table').any()
        # Check for textual
        assert ids.str.contains('average-emotion').any()
        # Check for propagation
        assert ids.str.contains('propagation').any()
        # Check for multimodal
        assert ids.str.contains('multimodal').any()

    def test_dataset_callbacks(self):
        # Outputs
        # When the dataset dropdown changes, the dataset storage should be updated
        self.assertIn('current-dataset-state', self.callbacks['dataset-dropdown-remiss-dashboard'])

        # Inputs
        # Control panel
        # When the dataset storage changes, the date range should be updated
        self.assertIn('date-picker-control-panel-remiss-dashboard', self.callbacks['current-dataset-state'])
        # When the dataset storage changes, the wordcloud should be updated
        self.assertIn('wordcloud-control-panel-remiss-dashboard', self.callbacks['current-dataset-state'])

        # Time series
        # When the dataset storage changes, the time series should be updated
        self.assertIn('fig-tweet-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-dataset-state'])
        self.assertIn('fig-users-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-dataset-state'])

        # Tweet table
        # When the dataset storage changes, the tweet table should be updated
        self.assertIn('table-tweet-table-remiss-dashboard', self.callbacks['current-dataset-state'])

        # Egonet
        # When the dataset storage changes, the egonet should be updated
        self.assertIn('fig-egonet-remiss-dashboard', self.callbacks['current-dataset-state'])

        # Other storage
        # When the dataset storage changes, the user should be cleared
        self.assertIn('current-user-state', self.callbacks['current-dataset-state'])
        # When the dataset storage changes, the hashtag should be cleared
        self.assertIn('current-hashtags-state', self.callbacks['current-dataset-state'])
        # When the dataset storage changes, the tweet should be cleared
        self.assertIn('current-tweet-state', self.callbacks['current-dataset-state'])

    def test_date_range_callbacks(self):
        # Outputs
        # When the date picker changes, the date storage should be updated
        self.assertIn('current-start-date-state', self.callbacks['date-picker-control-panel-remiss-dashboard'])
        self.assertIn('current-end-date-state', self.callbacks['date-picker-control-panel-remiss-dashboard'])

        # Inputs
        # Control panel
        # When the date range changes, the wordcloud should be updated
        self.assertIn('wordcloud-control-panel-remiss-dashboard', self.callbacks['current-start-date-state'])
        self.assertIn('wordcloud-control-panel-remiss-dashboard', self.callbacks['current-end-date-state'])

        # Time series
        # When the date range changes, the time series should be updated
        self.assertIn('fig-tweet-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-start-date-state'])
        self.assertIn('fig-users-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-start-date-state'])
        self.assertIn('fig-tweet-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-end-date-state'])
        self.assertIn('fig-users-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-end-date-state'])

        # Tweet table
        # When the date range changes, the tweet table should be updated
        self.assertIn('table-tweet-table-remiss-dashboard', self.callbacks['current-start-date-state'])
        self.assertIn('table-tweet-table-remiss-dashboard', self.callbacks['current-end-date-state'])

        # Egonet
        # When the date range changes, the egonet should be updated
        self.assertIn('fig-egonet-remiss-dashboard', self.callbacks['current-start-date-state'])
        self.assertIn('fig-egonet-remiss-dashboard', self.callbacks['current-end-date-state'])

    def test_user_callbacks(self):
        # Output
        # When tweet table user is selected the user storage should be updated
        self.assertIn('current-user-state', self.callbacks['table-tweet-table-remiss-dashboard'])
        # With a egonet node is clicked the user storage should be updated
        self.assertIn('current-user-state', self.callbacks['fig-egonet-remiss-dashboard'])

        # Inputs
        # When the user changes, egonet should be updated
        self.assertIn('fig-egonet-remiss-dashboard', self.callbacks['current-user-state'])

        # When the user changes, the profiling should be updated
        self.assertIn('fig-donut-plot-behaviour1-profiling-filterable-plots-remiss-dashboard',
                      self.callbacks['current-user-state'])

    def test_tweet_callbacks(self):
        # Output
        # When tweet table tweet is selected the tweet storage should be updated
        self.assertIn('current-tweet-state', self.callbacks['table-tweet-table-remiss-dashboard'])

        # Input
        # When the tweet storage changes, the propagation should be updated
        self.assertIn('fig-propagation-tree-propagation-filterable-plots-remiss-dashboard',
                      self.callbacks['current-tweet-state'])

        # When the tweet storage, the multimodal should be updated
        self.assertIn('multimodal-filterable-plots-remiss-dashboard-claim_text',
                      self.callbacks['current-tweet-state'])

    def test_hashtags_callbacks(self):
        # Outputs
        # When the wordcloud is clicked, the hashtag storage should be updated
        self.assertIn('current-hashtags-state', self.callbacks['wordcloud-control-panel-remiss-dashboard'])
        # When the tweet table text is selected the hashtag storage should be updated
        self.assertIn('current-hashtags-state', self.callbacks['table-tweet-table-remiss-dashboard'])

        # When the hashtag changes, the tweet table should be updated
        self.assertIn('table-tweet-table-remiss-dashboard', self.callbacks['current-hashtags-state'])

        # When the hashtag changes, the egonet should be updated
        self.assertIn('fig-egonet-remiss-dashboard', self.callbacks['current-hashtags-state'])

        # When the hashtag changes, the time series should be updated
        self.assertIn('fig-tweet-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-hashtags-state'])
        self.assertIn('fig-users-time-series-filterable-plots-remiss-dashboard',
                      self.callbacks['current-hashtags-state'])

    def get_ids(self, element):
        try:
            return [e['id'] for e in element]
        except TypeError:
            try:
                return [element.component_id]
            except AttributeError:
                return [e.component_id for e in element]


if __name__ == '__main__':
    unittest.main()
