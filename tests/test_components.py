import unittest
from contextvars import copy_context
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

from dash._callback_context import context_value
from dash._utils import AttributeDict
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
        self.egonet_plot_factory = Mock()
        self.egonet_plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.egonet_plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                                datetime(2023, 12, 31))
        self.egonet_plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.top_table_factory = Mock()
        self.top_table_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.top_table_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                              datetime(2023, 12, 31))
        self.top_table_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.top_table_factory.retweeted_table_columns = ['id', 'text', 'user']
        self.top_table_factory.user_table_columns = ['id', 'name', 'screen_name']
        self.top_table_factory.top_table_columns = ['User', 'Text', 'Retweets', 'Is usual suspect', 'Party']

        self.uv_factory = Mock()
        self.uv_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.uv_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                        datetime(2023, 12, 31))
        self.uv_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.uv_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                        ('hashtag3', 3), ('hashtag4', 2),
                                                        ('hashtag5', 1), ('hashtag6', 1), ]

        self.component = RemissDashboard(self.tweet_user_plot_factory, self.top_table_factory, self.egonet_plot_factory, self.uv_factory)

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a Dropdown
        # - a TweetUserTimeSeriesComponent
        # - a EgonetComponent
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                if component.children is not None:
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
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn(f'dataset-dropdown', component_ids)
        self.assertIn('date-picker', component_ids)
        self.assertIn('wordcloud', component_ids)
        self.assertIn('fig-tweet', component_ids)
        self.assertIn('fig-users', component_ids)
        self.assertIn('user-dropdown', component_ids)
        self.assertIn('slider', component_ids)
        self.assertIn('fig', component_ids)

if __name__ == '__main__':
    unittest.main()
