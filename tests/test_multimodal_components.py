import unittest
import uuid
from unittest import TestCase

import dash_bootstrap_components as dbc
from dash import Dash, Output, Input
from dash import dcc
from dash.dcc import Graph
from pymongo import MongoClient

from components.dashboard import RemissState
from components.multimodal import MultimodalComponent
from figures.multimodal import MultimodalPlotFactory


class TestMultimodalComponentComponent(TestCase):
    def setUp(self):
        self.plot_factory = MultimodalPlotFactory(data_dir='./../fact_checking_data')
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database(self.tmp_dataset)
        self.database = self.client.get_database(self.tmp_dataset)
        self.collection = self.database.get_collection('multimodal')
        self.test_data = [
            {
                "visual_evidence_domain": "elpais.com",
                "visual_evidence_matched_categories": "['caption', 'place', 'vit', 'objects']",
                "visual_evidence_text": " El candidato a lehendakari del PNV, Imanol Pradales, comparece en la sede del PNV tras el conteo, este domingo.Resultados de las elecciones vascas 2024 | El PNV empata con Bildu y podr\u00e1 reeditar la coalici\u00f3n con los socialistas | Elecciones en el Pa\u00eds Vasco 21-A | EL PA\u00cdS",
                "visual_evidence_similarity_score": "0.533111073076725",
                "visual_evidence_graph_similarity_score": "1.0",
                "text_evidence": "no_text_evidence",
                "text_evidence_similarity_score": "0.4457797110080719",
                "text_evidence_graph_similarity_score": "0.0",
                "visual_evidence_domain1": "elpais.com",
                "visual_evidence_matched_categories1": "['caption', 'place', 'vit', 'objects']",
                "visual_evidence_text1": " El candidato a lehendakari del PNV, Imanol Pradales, comparece en la sede del PNV tras el conteo, este domingo.Resultados de las elecciones vascas 2024 | El PNV empata con Bildu y podr\u00e1 reeditar la coalici\u00f3n con los socialistas | Elecciones en el Pa\u00eds Vasco 21-A | EL PA\u00cdS",
                "visual_evidence_similarity_score1": "0.5365476682782173",
                "visual_evidence_graph_similarity_score1": "1.0",
                "claim_text": "Desde hace tiempo se ve\u00eda venir que se pod\u00eda vertebrar una coalici\u00f3n como en Euskadi @eajpnv + @socialistavasco en Nafarroa @PSNPSOE + @geroabai \nCon estas declaraciones queda claro que esa es la estrategia nacional del PNV en toda Euskal Herria.\n#26M https://t.co/nAOIdpq1YB",
                "found_flag": "not found",
                "id_in_json": 7620,
                "t_sug": "Estrategia Nacional PNV Euskal Herria EAJPNV SocialistaVasco Navarra",
                "old_t_sug": "coalici\u00f3n Euskadi Euskal Herria estrategia nacional PNV",
                "results": {
                    "predicted_label": "FAKE",
                    "actual_label": "FAKE",
                    "visual_similarity_score": 0.533111073076725,
                    "explanations": ""
                },
                "tweet_id": "1133352119124353024"
            },
        ]
        self.collection.insert_many(self.test_data)
        self.client.close()

        self.state = RemissState(name='state')
        self.component = MultimodalComponent(self.plot_factory, self.state, name='multimodal')

    def tearDown(self) -> None:
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database(self.tmp_dataset)
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
        expected_components = ['fig-claim-image', 'fig-graph-claim', 'fig-evidence-image-1', 'fig-graph-evidence-text',
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
            if 'MultimodalComponent.update' in str(cb["callback"]):
                callback = cb
                break

        self.assertEqual(callback['inputs'], [{'id': 'current-dataset-state', 'property': 'data'},
                                              {'id': 'current-tweet-state', 'property': 'data'}])
        actual_output = [{'component_id': o.component_id, 'property': o.component_property} for o in
                         callback['output']]
        self.assertEqual(actual_output, [{'component_id': 'fig-claim-image-multimodal', 'property': 'figure'},
                                         {'component_id': 'fig-evidence-image-multimodal', 'property': 'figure'},
                                         {'component_id': 'fig-graph-claim-multimodal', 'property': 'figure'},
                                         {'component_id': 'fig-graph-evidence-text-multimodal', 'property': 'figure'},
                                         {'component_id': 'fig-graph-evidence-vis-multimodal', 'property': 'figure'},
                                         {'component_id': 'fig-evidence-image-1-multimodal', 'property': 'figure'},
                                         {'component_id': 'claim-text-multimodal', 'property': 'children'},
                                         {'component_id': 'text-evidences-multimodal', 'property': 'children'},
                                         {'component_id': 'evidence-text-multimodal', 'property': 'children'},
                                         {'component_id': 'evidence-image-alt-text-multimodal', 'property': 'children'},
                                         {'component_id': 'predicted-label-multimodal', 'property': 'children'},
                                         {'component_id': 'actual-label-multimodal', 'property': 'children'},
                                         {'component_id': 'num-claim-edges-multimodal', 'property': 'children'},
                                         {'component_id': 'frac-verified-multimodal', 'property': 'children'},
                                         {'component_id': 'explanations-multimodal', 'property': 'children'},
                                         {'component_id': 'visual-similarity-score-multimodal', 'property': 'children'},
                                         {'component_id': 'collapse-multimodal', 'property': 'is_open'}])


if __name__ == '__main__':
    unittest.main()
