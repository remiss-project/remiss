from unittest import TestCase

from dash.html import Iframe
from pymongo import MongoClient

from components.dashboard import RemissState
from components.fact_checking import FactCheckingComponent
from figures.fact_checking import FactCheckingPlotFactory


class FactCheckingComponentTest(TestCase):
    def setUp(self):
        self.plot_factory = FactCheckingPlotFactory(data_dir='./../fact_checking_data')
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('fact_checking')
        self.database = self.client.get_database('fact_checking')
        self.collection = self.database.get_collection('test_dataset')
        test_data = [{"id": 47, 'tweet_id': '100485425'},
                     {'id': 67, 'tweet_id': '100485426'}]
        self.collection.insert_many(test_data)

        self.state = RemissState(name='state')
        self.component = FactCheckingComponent(self.plot_factory, self.state, name='fact_checking')

    def test_layout(self):
        layout = self.component.layout()

        # check that among the final components of the layout we have:
        # - a DatePickerRange
        # - a Wordcloud
        # - a TimeSeriesComponent
        # find components recursively
        def find_components(component, found_components):
            if hasattr(component, 'children'):
                for child in component.children:
                    find_components(child, found_components)
            if isinstance(component, Iframe):
                found_components.append(component)

        found_components = []
        find_components(layout, found_components)
        found_components = [type(component) for component in found_components]
        self.assertIn(Iframe, found_components)

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
        component_ids = ['-'.join(component.id.split('-')[:-1]) for component in found_components]
        self.assertIn('fig-tweet', component_ids)
        self.assertIn('fig-users', component_ids)
        found_main_ids = ['-'.join(component.id.split('-')[-1:]) for component in found_components]
        self.assertIn(self.component.name, found_main_ids)
        self.assertEqual(len(set(found_main_ids)), 1)

    def test_update_callback(self):
        app = Dash()
        self.component.callbacks(app)

        callback = None
        for cb in app.callback_map.values():
            if 'TimeSeriesComponent.update' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'current-dataset-state', 'property': 'data'},
                                              {'id': 'current-hashtags-state', 'property': 'data'},
                                              {'id': 'current-start-date-state', 'property': 'data'},
                                              {'id': 'current-end-date-state', 'property': 'data'}])
        actual_output = [{'component_id': o.component_id, 'property': o.component_property} for o in
                         callback['output']]
        self.assertEqual(actual_output, [{'component_id': 'fig-tweet-timeseries', 'property': 'figure'},
                                         {'component_id': 'fig-users-timeseries', 'property': 'figure'}])


if __name__ == '__main__':
    unittest.main()
