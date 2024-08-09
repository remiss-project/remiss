import unittest
from contextvars import copy_context
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

import dash_bootstrap_components as dbc
from dash import Dash, dcc, Output, Input
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.dcc import Graph, Dropdown, Slider

from components.dashboard import RemissState
from components.egonet import EgonetComponent
from figures.propagation import PropagationPlotFactory


class EgonetComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = Mock()
        self.plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                         datetime(2023, 12, 31))
        self.plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.plot_factory.plot_egonet = Mock()
        self.plot_factory.plot_hidden_network = Mock()
        self.state = RemissState(name='state')
        self.component = EgonetComponent(self.plot_factory, self.state, name='egonet')
        self.test_dataset = 'test_dataset_2'
        self.test_user_id = '377541658'
        self.test_tweet_id = '1167078759280889856'

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a Dropdown
        # - a Slider
        # - a graph
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                if len(component.children) == 0:
                    find_components(component.children, found_components)
                else:
                    for child in component.children:
                        find_components(child, found_components)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Slider, found_components)
        self.assertIn(Graph, found_components)

    def test_layout_ids(self):
        layout = self.component.layout()

        # check that among the ids are correctly patched
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
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

        found_components = []
        find_components(layout, found_components)
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn('slider', component_ids)
        self.assertIn('fig', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def test_update_user_state_user_show_egonet(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "current-user-state.data"}]}))
            return self.component.update('test_dataset', 'test_user_egonet',
                                         '2023-01-01',
                                         '2023-12-31',
                                         ['test_hashtags'],
                                         1)

        self.component.dates = ['2023-01-01', '2023-12-31']
        ctx = copy_context()
        output = ctx.run(run_callback)
        self.plot_factory.plot_egonet.assert_called_once_with(
            'test_dataset', 'test_user_egonet', 1, '2023-01-01', '2023-12-31', ['test_hashtags'])

    def test_update_user_state_user_show_hidden_network(self):
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "current-user-state.data"}]}))
            return self.component.update('test_dataset', 'test_user_egonet',
                                         '2023-01-01',
                                         '2023-12-31',
                                         ['test_hashtags'],
                                         1)

        self.component.plot_factory.plot_egonet.side_effect = RuntimeError('User not available')
        self.component.dates = ['2023-01-01', '2023-12-31']
        ctx = copy_context()
        output = ctx.run(run_callback)
        self.plot_factory.plot_egonet.assert_called_once_with(
            'test_dataset', 'test_user_egonet', 1, '2023-01-01', '2023-12-31', ['test_hashtags'])
        self.plot_factory.plot_hidden_network.assert_called_once_with('test_dataset',
                                                                      start_date='2023-01-01',
                                                                      end_date='2023-12-31',
                                                                      hashtag=['test_hashtags'])

    @unittest.skip("Render")
    def test_render(self):
        plot_factory = PropagationPlotFactory()
        component = EgonetComponent(plot_factory, self.state, name='egonet')
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
        component.callbacks(app)
        self.state.callbacks(app)
        dataset_dropdown = dcc.Dropdown(id='dataset-dropdown',
                                        options=[{'label': 'test_dataset', 'value': self.test_dataset}],
                                        value=self.test_dataset)
        user_dropdown = dcc.Dropdown(id='tweet-dropdown',
                                     options=[{'label': self.test_user_id, 'value': self.test_user_id},
                                              {'label': 'POTATO', 'value': 'POTATO'}],
                                     value=self.test_user_id)

        def update(x):
            return x

        app.callback(Output(self.state.current_dataset, 'data'),
                     [Input('dataset-dropdown', 'value')])(update)
        app.callback(Output(self.state.current_user, 'data'),
                     [Input('tweet-dropdown', 'value')])(update)
        app.layout = dbc.Container([
            self.state.layout(),
            dbc.Row([
                dbc.Col([
                    component.layout()
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dataset_dropdown
                ]),
                dbc.Col([
                    user_dropdown
                ]),
            ]),
        ])

        app.run_server(debug=True, port=8050)


if __name__ == '__main__':
    unittest.main()
