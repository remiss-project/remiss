import unittest
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

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

        self.component = RemissDashboard(
            self.tweet_user_plot_factory,
            self.tweet_table_factory,
            self.propagation_factory,
            self.textual_factory,
            self.profile_factory,
            self.multimodal_factory)

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a Dropdown
        # - a TweetUserTimeSeriesComponent
        # - a EgonetComponent
        # find components recursively
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

        # check that among the ids are correctly patched
        # find components recursively
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
        actual_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        expected_ids = ['dataset-dropdown', 'fig-cascade-ccdf-cascade-ccdf-general-plots',
                        'fig-cascade-count-over-time-cascade-count-over-time-general-plots',
                        'fig-average-emotion-general-plots', 'fig-emotion-per-hour-general-plots',
                        'date-picker-control-panel', 'wordcloud-control-panel', 'fig-egonet', 'user-dropdown-egonet',
                        'slider-egonet', 'date-slider-egonet', 'fig-tweet-time-series-filterable-plots',
                        'fig-users-time-series-filterable-plots', 'fig-radarplot-emotions-profiling-filterable-plots',
                        'fig-vertical-barplot-polarity-profiling-filterable-plots',
                        'fig-donut-plot-behaviour1-profiling-filterable-plots',
                        'fig-donut-plot-behaviour2-profiling-filterable-plots',
                        'fig-claim-image-multimodal-filterable-plots', 'fig-graph-claim-multimodal-filterable-plots',
                        'fig-visual-evidences-multimodal-filterable-plots',
                        'fig-graph-evidence-text-multimodal-filterable-plots',
                        'fig-evidence-image-multimodal-filterable-plots',
                        'fig-graph-evidence-vis-multimodal-filterable-plots',
                        'fig-propagation-tree-propagation-filterable-plots',
                        'fig-propagation-depth-propagation-filterable-plots',
                        'fig-propagation-size-propagation-filterable-plots',
                        'fig-propagation-max-breadth-propagation-filterable-plots',
                        'fig-propagation-structural-virality-propagation-filterable-plots']

        self.assertEqual(set(actual_ids), set(expected_ids))


if __name__ == '__main__':
    unittest.main()
