import unittest
from unittest import TestCase

import dash_bootstrap_components as dbc
from dash import Dash, Output, Input
from dash import dcc
from dash.dcc import Graph
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
        self.test_data = [
            {
                "claim_text": "A shot of the All Nippon Airways Boeing 787 Dreamliner that s painted in the likeness of R2D2 in Los Angeles on Dec 15 2015",
                "id": 47,
                "tweet_id": "100485425",
                "text_evidences": "-\n ANA's R2D2 Jet Uses The Force to Transport Stars Between The 'Star \nWars' Premieres - TheDesignAir\n\n- The Cast Of \"Star Wars: The Force Awakens\" On ANA Charter Flight From \nLos Angeles To The London Premiere\n\n- The R2-D2 ANA Jet Transports Star Wars Movie Cast Between Premieres in\n USA and UK\n\n- Dec15.32\n\n- 24 Boeing 787 ideas | boeing 787, boeing, boeing 787 ... - Pinterest\n\n- The stars of \"Star Wars: The Force Awakens\" blew into London in It \nMovie Cast, It Cast, Geek Movies, Star Wars Cast, Private Pilot, Air \nPhoto, Airplane Design, Aircraft Painting, Commercial Aircraft\n\n- 19 Geek Stuff ideas | geek stuff, star wars, stars\n\n- 100 Aviation ideas | aviation, boeing, aircraft\n",
                "evidence_text": "The Cast Of \"Star Wars: The Force Awakens\" On ANA Charter Flight From Los Angeles To The London Premiere",
                "evidence_image_alt_text": "Page\n 2 - R2d2 Star Wars High Resolution Stock Photography and Images - Alamy\n Page 2 - R2d2 Star Wars High Resolution Stock Photography and ...",
                "results": {
                    "predicted_label": 1,
                    "actual_label": 0,
                    "num_claim_edges": 5,
                    "frac_verified": 0.0,
                    "explanations": "+ XT(V) ns + XV(T) ns",
                    "visual_similarity_score": 0.8824891924858094
                }
            }
        ]
        self.collection.insert_many(self.test_data)
        self.client.close()

        self.state = RemissState(name='state')
        self.component = FactCheckingComponent(self.plot_factory, self.state, name='fact_checking')

    def tearDown(self) -> None:
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('fact_checking')
        self.client.close()

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
        expected_components = ['fig-claim-image', 'fig-graph-claim', 'fig-visual-evidences', 'fig-graph-evidence-text',
                               'fig-evidence-image', 'fig-graph-evidence-vis']
        self.assertEqual(set(component_ids), set(expected_components))
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
                                      options=[{'label': '100485425', 'value': '100485425'},
                                               {'label': '100485426', 'value': '100485426'}],
                                      value='100485425')

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
