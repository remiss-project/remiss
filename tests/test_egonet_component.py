import unittest
from contextvars import copy_context
from datetime import datetime
from unittest import TestCase
from unittest.mock import Mock

from dash import Dash
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.dcc import Graph, Dropdown, Slider

from components.dashboard import RemissState
from components.egonet import EgonetComponent


class EgonetComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = Mock()
        self.plot_factory.available_datasets = ['dataset1', 'dataset2', 'dataset3']
        self.plot_factory.get_date_range.return_value = (datetime(2023, 1, 1),
                                                         datetime(2023, 12, 31))
        self.plot_factory.get_users.return_value = ['user1', 'user2', 'user3']
        self.state = RemissState(name='state')
        self.component = EgonetComponent(self.plot_factory, self.state, name='egonet')

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
            if isinstance(component, Dropdown):
                found_components.append(component)
            if isinstance(component, Slider):
                found_components.append(component)
            if isinstance(component, Graph):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Dropdown, found_components)
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
        self.assertIn('user-dropdown', component_ids)
        self.assertIn('slider', component_ids)
        self.assertIn('fig', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    # TODO implement those tests
    # def test_get_egonet_missing_user_backbone(self):
    #     user_id = '1'
    #     depth = 2
    #     self.egonet.threshold = 0.4
    #
    #     actual = self.egonet.get_egonet(self.test_dataset, user_id, depth)
    #
    #     # check it returns the hidden network backbone
    #     self.assertEqual(actual.vcount(), 3224)
    #     self.assertEqual(actual.ecount(), 4801)

    # def test_get_egonet_missing_user_full(self):
    #     user_id = '1'
    #     depth = 2
    #
    #     actual = self.egonet.get_egonet(self.test_dataset, user_id, depth)
    #
    #     # check it returns the full hidden network
    #     self.assertEqual(actual.vcount(), 3315)
    #     self.assertEqual(actual.ecount(), 5844)


    def test_update_egonet_callback(self):
        app = Dash(prevent_initial_callbacks="initial_duplicate",
                   meta_tags=[
                       {
                           "name": "viewport",
                           "content": "width=device-width, initial-scale=1, maximum-scale=1",
                       }
                   ],
                   )
        self.component.callbacks(app)

        # Simulate the update function for the plots
        plots_key = f'fig-{self.component.name}.figure'
        callback = app.callback_map[plots_key]
        expected = [{'id': 'current-dataset-state', 'property': 'data'},
                    {'id': 'current-user-state', 'property': 'data'},
                    {'id': 'current-start-date-state', 'property': 'data'},
                    {'id': 'current-end-date-state', 'property': 'data'},
                    {'id': 'current-hashtags-state', 'property': 'data'},
                    {'id': 'slider-egonet', 'property': 'value'},
                    {'id': 'date-slider-egonet', 'property': 'value'},
                    {'id': 'user-dropdown-egonet', 'property': 'disabled'},
                    {'id': 'slider-egonet', 'property': 'disabled'},
                    {'id': 'date-slider-egonet', 'property': 'disabled'}]
        self.assertEqual(expected, callback['inputs'])
        self.assertEqual(callback['output'].component_id, f'fig-{self.component.name}')
        self.assertEqual(callback['output'].component_property, 'figure')


if __name__ == '__main__':
    unittest.main()
