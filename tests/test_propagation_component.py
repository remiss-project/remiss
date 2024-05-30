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

    @classmethod
    def setUpClass(cls):
        populate_test_database('test_dataset')

    @classmethod
    def tearDownClass(cls):
        delete_test_database('test_dataset')

    def setUp(self):
        self.plot_factory = PropagationPlotFactory()
        self.state = RemissState(name='state')
        self.component = PropagationComponent(self.plot_factory, self.state, name='propagation')

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
        expected_components = {'fig-cascade-ccdf',
                               'fig-cascade-count-over-time',
                               'fig-propagation-depth',
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
                                        options=[{'label': 'test_dataset', 'value': 'test_dataset'}],
                                        value='test_dataset')
        tweet_dropdown = dcc.Dropdown(id='tweet-dropdown',
                                      options=[{'label': '1160842257647493120', 'value': '1160842257647493120'},
                                               {'label': '1160842257647493120', 'value': '1160842257647493120'}],
                                      value='1160842257647493120')

        app.callback(Output(self.state.current_dataset, 'data'),
                     [Input('dataset-dropdown', 'value')])(lambda x: x)
        app.callback(Output(self.state.current_tweet, 'data'),
                     [Input('tweet-dropdown', 'value')])(lambda x: x)
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
                                          'property': 'figure'}])

    def test_update_cascade_callback(self):
        app = Dash()
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'PropagationComponent.update_cascade' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'current-dataset-state', 'property': 'data'}])
        actual_output = [{'component_id': o.component_id, 'property': o.component_property} for o in
                         callback['output']]
        self.assertEqual(actual_output, [{'component_id': 'fig-cascade-ccdf-propagation', 'property': 'figure'},
                                         {'component_id': 'fig-cascade-count-over-time-propagation',
                                          'property': 'figure'}])


if __name__ == '__main__':
    unittest.main()