import unittest
from contextvars import copy_context
from datetime import datetime
from unittest.mock import Mock

from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.dcc import Dropdown, Slider, Graph, DatePickerRange
from dash.exceptions import PreventUpdate
from dash_holoniq_wordcloud import DashWordcloud

from components.dashboard import ControlPanelComponent, RemissState
from tests.conftest import find_components


class MyTestCase(unittest.TestCase):
    def setUp(self):
        plot_factory = Mock()
        plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                    datetime(2023, 12, 31))
        plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        plot_factory.get_hashtag_freqs.return_value = [('hashtag1', 10), ('hashtag2', 5),
                                                       ('hashtag3', 3), ('hashtag4', 2),
                                                       ('hashtag5', 1), ('hashtag6', 1), ]
        state = RemissState(name='state')
        max_wordcloud_words = 100
        wordcloud_width = 800
        wordcloud_height = 400
        match_wordcloud_width = False

        self.component = ControlPanelComponent(plot_factory,
                                               state=state,
                                               name='control',
                                               max_wordcloud_words=max_wordcloud_words,
                                               wordcloud_width=wordcloud_width,
                                               wordcloud_height=wordcloud_height,
                                               match_wordcloud_width=match_wordcloud_width)

    def test_dataset_storage(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "dataset-dropdown"}))
            actual = self.component.update_dataset_storage('dataset2')
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual(actual, 'dataset2')

    def test_date_range(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "date-picker"}))
            actual = self.component.update_date_range('dataset2')
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual(actual, (datetime(2023, 1, 1), datetime(2023, 12, 31),
                                  datetime(2023, 1, 1), datetime(2023, 12, 31, )))

    def test_wordcloud(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "dataset-dropdown"}))
            actual = self.component.update_wordcloud('dataset2', 'user1', '12/1/2023', '12/31/2023')
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual(actual, ([('hashtag1', 10), ('hashtag2', 5),
                                   ('hashtag3', 3), ('hashtag4', 2),
                                   ('hashtag5', 1), ('hashtag6', 1)], 10))

    def test_hashtags(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "wordcloud"}))
            actual = self.component.update_hashtag_storage(click_data=['hashtag1', 58.205296844244])
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual((['hashtag1'], None, None), actual)

    def test_hashtags2(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "wordcloud"}))
            actual = self.component.update_hashtag_storage(click_data=[])
            return actual

        ctx = copy_context()
        with self.assertRaises(PreventUpdate):
            ctx.run(run_callback)


    def test_start_date(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "date-picker"}))
            actual = self.component.update_start_date_storage(datetime(2023, 1, 2))
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual(actual, datetime(2023, 1, 2))

    def test_end_date(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_id": "date-picker"}))
            actual = self.component.update_end_date_storage(datetime(2023, 1, 2))
            return actual

        ctx = copy_context()
        actual = ctx.run(run_callback)
        self.assertEqual(actual, datetime(2023, 1, 2))

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a Dropdown
        # - a TweetUserTimeSeriesComponent
        # - a EgonetComponent
        # find components recursively
        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(DashWordcloud, found_components)
        self.assertIn(DatePickerRange, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn('date-picker', component_ids)
        self.assertIn('wordcloud', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)



if __name__ == '__main__':
    unittest.main()
