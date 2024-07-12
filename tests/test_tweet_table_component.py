import unittest
import uuid

from datetime import datetime
from unittest import TestCase

from dash import Dash
from pymongo import MongoClient
import pandas as pd

from components.control_panel import ControlPanelComponent
from components.dashboard import RemissState
from components.tweet_table import TweetTableComponent
from figures import TweetTableFactory
import dash_bootstrap_components as dbc


class TestTopTableComponent(TestCase):
    def setUp(self):
        self.top_table_factory = TweetTableFactory()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database(self.tmp_dataset)
        self.database = self.client.get_database(self.tmp_dataset)
        self.collection = self.database.get_collection('raw')
        test_data = [{"id": '0', "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                      'text': 'test_text',
                      'public_metrics': {'retweet_count': 1, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_0", "id": '0',
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': False,
                                                     'has_profiling': False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": '1', "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      'text': 'test_text2',
                      'public_metrics': {'retweet_count': 2, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_1", "id": '1',
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False,
                                                     'has_multimodal_fact-checking': True,
                                                     'has_profiling': False
                                                     }},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": '1', "type": "quoted"}]},
                     {"id": '2', "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      'text': 'test_text3',
                      'public_metrics': {'retweet_count': 3, 'reply_count': 2, 'like_count': 3, 'quote_count': 4},
                      "author": {"username": "TEST_USER_2", "id": '2',
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True,
                                                     }},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": '1', "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

        multimodal_collection = self.database.get_collection('multimodal')
        multimodal_data = [{'tweet_id': '0'}, {'tweet_id': '2'}]
        multimodal_collection.insert_many(multimodal_data)

        profiling_collection = self.database.get_collection('profiling')
        profiling_data = [{'user_id': '0'}, {'user_id': '1'}]
        profiling_collection.insert_many(profiling_data)

        textual = self.database.get_collection('textual')
        textual_data = [{'tweet_id': '0', 'fakeness_probabilities': 0.1},
                        {'tweet_id': '1', 'fakeness_probabilities': 0.2},
                        {'tweet_id': '2', 'fakeness_probabilities': 0.3}]
        textual.insert_many(textual_data)

        self.state = RemissState(name='state')

        self.component = TweetTableComponent(self.top_table_factory, self.state, name='table_test')

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_update_callback(self):
        app = Dash(prevent_initial_callbacks='initial_duplicate')
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'TweetTableComponent.update' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'current-dataset-state', 'property': 'data'},
                                              {'id': 'current-start-date-state', 'property': 'data'},
                                              {'id': 'current-end-date-state', 'property': 'data'},
                                              ])
        actual_output = callback['output']
        self.assertEqual(actual_output.component_id, 'table-table_test')
        self.assertEqual(actual_output.component_property, 'data')

    def test_update_user_callback(self):
        app = Dash(prevent_initial_callbacks='initial_duplicate')
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'TweetTableComponent.update_user' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'table-table_test', 'property': 'selected_rows'}])
        actual_output = callback['output']
        self.assertEqual(actual_output.component_id, 'current-user-state')
        self.assertEqual(actual_output.component_property, 'data')

    def test_update_tweet_callback(self):
        app = Dash(prevent_initial_callbacks='initial_duplicate')
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'TweetTableComponent.update_tweet' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'table-table_test', 'property': 'selected_rows'}])
        actual_output = callback['output']
        self.assertEqual(actual_output.component_id, 'current-tweet-state')
        self.assertEqual(actual_output.component_property, 'data')

    def test_update_hashtags_callback(self):
        app = Dash(prevent_initial_callbacks='initial_duplicate')
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'TweetTableComponent.update_hashtags' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'table-table_test', 'property': 'selected_rows'}])
        actual_output = callback['output']
        self.assertEqual(actual_output.component_id, 'current-hashtags-state')
        self.assertEqual(actual_output.component_property, 'data')

    def _test_render(self):
        dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

        control_panel = ControlPanelComponent(self.top_table_factory, self.state)

        app = Dash(prevent_initial_callbacks='initial_duplicate',
                   external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc_css], )
        self.component.callbacks(app)
        control_panel.callbacks(app)

        app.layout = dbc.Container([
            self.state.layout(),
            control_panel.layout(),
            self.component.layout()
        ])
        app.run(debug=True)
