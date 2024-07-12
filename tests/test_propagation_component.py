import unittest

import dash_bootstrap_components as dbc
from dash import Dash, Output, Input
from dash import dcc
from dash.dcc import Graph
from pymongo import MongoClient

from components.dashboard import RemissState
from components.propagation import PropagationComponent
from figures.propagation import PropagationPlotFactory
from tests.conftest import populate_test_database, delete_test_database


class PropagationComponentTest(unittest.TestCase):

    def setUp(self):
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.test_user_id = '1033714286231740416'
        self.test_tweet_id = '1167078759280889856'
        self.plot_factory = PropagationPlotFactory(available_datasets=[self.test_dataset])
        self.state = RemissState(name='state')
        self.component = PropagationComponent(self.plot_factory, self.state, name='propagation')

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children') and component.children:
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        expected_components = {'fig-propagation-depth',
                               'fig-propagation-max-breadth',
                               'fig-propagation-size',
                               'fig-propagation-structural-virality',
                               'fig-propagation-tree'}
        self.assertEqual(set(component_ids), expected_components)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def _test_render(self):
        dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
        app = Dash(__name__,
                   external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc_css],
                   prevent_initial_callbacks="initial_duplicate",
                   meta_tags=[
                       {
                           "name": "viewport",
                           "content": "width=device-width, initial-scale=1, maximum-scale=1",
                       }
                   ],
                   )
        self.component.callbacks(app)
        self.state.callbacks(app)
        dataset_dropdown = dcc.Dropdown(id='dataset-dropdown',
                                        options=[{'label': 'test_dataset', 'value': self.test_dataset}],
                                        value=self.test_dataset)
        tweet_dropdown = dcc.Dropdown(id='tweet-dropdown',
                                      options=[{'label': self.test_tweet_id, 'value': self.test_tweet_id},
                                               {'label': 'POTATO', 'value': 'POTATO'}],
                                      value=self.test_tweet_id)

        def update(x):
            return x

        app.callback(Output(self.state.current_dataset, 'data'),
                     [Input('dataset-dropdown', 'value')])(update)
        app.callback(Output(self.state.current_tweet, 'data'),
                     [Input('tweet-dropdown', 'value')])(update)
        app.layout = dbc.Container([
            self.state.layout(),
            self.component.layout(),
            dataset_dropdown,
            tweet_dropdown
        ])

        app.run_server(debug=True, port=8050)

    def test_update_tweet_callback(self):
        app = Dash()
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'PropagationComponent.update_tweet' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'current-dataset-state', 'property': 'data'},
                                              {'id': 'current-tweet-state', 'property': 'data'}])
        actual_output = [{'component_id': o.component_id, 'property': o.component_property} for o in
                         callback['output']]
        self.assertEqual(actual_output, [{'component_id': 'fig-propagation-tree-propagation', 'property': 'figure'},
                                         {'component_id': 'fig-propagation-depth-propagation', 'property': 'figure'},
                                         {'component_id': 'fig-propagation-size-propagation', 'property': 'figure'},
                                         {'component_id': 'fig-propagation-max-breadth-propagation',
                                          'property': 'figure'},
                                         {'component_id': 'fig-propagation-structural-virality-propagation',
                                          'property': 'figure'},
                                         {'component_id': 'collapse-propagation', 'property': 'is_open'}
                                         ])


if __name__ == '__main__':
    unittest.main()
