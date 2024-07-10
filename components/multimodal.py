from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input
from components.components import RemissComponent


class MultimodalComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.claim_image = dcc.Graph(figure={}, id=f'fig-claim-image-{self.name}')
        self.evidence_image = dcc.Graph(figure={}, id=f'fig-evidence-image-{self.name}')

        self.graph_claim = dcc.Graph(figure={}, id=f'fig-graph-claim-{self.name}')
        self.graph_evidence_text = dcc.Graph(figure={}, id=f'fig-graph-evidence-text-{self.name}')
        self.graph_evidence_vis = dcc.Graph(figure={}, id=f'fig-graph-evidence-vis-{self.name}')
        self.visual_evidences = dcc.Graph(figure={}, id=f'fig-visual-evidences-{self.name}')

        self.claim_text = html.P()
        self.text_evidences = html.P()
        self.evidence_text = html.P()
        self.evidence_image_alt_text = html.P()
        self.predicted_label = html.P()
        self.actual_label = html.P()
        self.num_claim_edges = html.P()
        self.frac_verified = html.P()
        self.explanations = html.P()
        self.visual_similarity_score = html.P()

        self.plot_factory = plot_factory
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                html.H3('Claim'),
                dbc.Card([
                    dbc.CardHeader('Claim Image'),
                    dbc.CardBody([
                        self.claim_image
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Claim Text'),
                    dbc.CardBody([
                        self.claim_text
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Graph Claim'),
                    dbc.CardBody([
                        self.graph_claim
                    ])
                ]),
            ]),
            dbc.Col([
                html.H3('Evidences'),
                dbc.Card([
                    dbc.CardHeader('Text Evidences'),
                    dbc.CardBody([
                        self.text_evidences
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Visual Evidences'),
                    dbc.CardBody([
                        self.visual_evidences
                    ])
                ]),
            ]),
            dbc.Col([
                html.H3('Results'),
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader('Predicted Label'),
                                    dbc.CardBody([
                                        self.predicted_label
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('frac. verified'),
                                    dbc.CardBody([
                                        self.frac_verified
                                    ])
                                ]),
                            ]),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader('Actual Label'),
                                    dbc.CardBody([
                                        self.actual_label
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Explanations'),
                                    dbc.CardBody([
                                        self.explanations
                                    ])
                                ]),
                            ]),
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader('Number of Claim Edges'),
                                    dbc.CardBody([
                                        self.num_claim_edges
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Similarity Score'),
                                    dbc.CardBody([
                                        self.visual_similarity_score
                                    ])
                                ]),
                            ]),
                        ]),
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Evidence Text'),
                    dbc.CardBody([
                        self.evidence_text
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Graph Evidence Text'),
                    dbc.CardBody([
                        self.graph_evidence_text
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Evidence Image'),
                    dbc.CardBody([
                        self.evidence_image
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Evidence Image Alt Text'),
                    dbc.CardBody([
                        self.evidence_image_alt_text
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Graph Evidence Visualization'),
                    dbc.CardBody([
                        self.graph_evidence_vis
                    ])
                ]),
            ])
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, tweet_id):
        try:
            metadata = self.plot_factory.get_user_metadata(dataset, tweet_id)
            claim_text = metadata['claim_text']
            text_evidences = metadata['text_evidences']
            evidence_text = metadata['evidence_text']
            evidence_image_alt_text = metadata['evidence_image_alt_text']
            predicted_label = metadata['results']['predicted_label']
            actual_label = metadata['results']['actual_label']
            num_claim_edges = metadata['results']['num_claim_edges']
            frac_verified = metadata['results']['frac_verified']
            explanations = metadata['results']['explanations']
            visual_similarity_score = metadata['results']['visual_similarity_score']

            claim_image = self.plot_factory.plot_claim_image(dataset, tweet_id)
            evidence_image = self.plot_factory.plot_evidence_image(dataset, tweet_id)
            graph_claim = self.plot_factory.plot_graph_claim(dataset, tweet_id)
            graph_evidence_text = self.plot_factory.plot_graph_evidence_text(dataset, tweet_id)
            graph_evidence_vis = self.plot_factory.plot_graph_evidence_vis(dataset, tweet_id)
            visual_evidences = self.plot_factory.plot_visual_evidences(dataset, tweet_id)

            return claim_image, evidence_image, graph_claim, graph_evidence_text, graph_evidence_vis, visual_evidences, \
                claim_text, text_evidences, evidence_text, evidence_image_alt_text, predicted_label, actual_label, \
                num_claim_edges, frac_verified, explanations, visual_similarity_score
        except RuntimeError as e:
            return {}, {}, {}, {}, {}, {}, '', '', '', '', '', '', '', '', '',''

    def callbacks(self, app):
        app.callback(
            Output(self.claim_image, 'figure'),
            Output(self.evidence_image, 'figure'),
            Output(self.graph_claim, 'figure'),
            Output(self.graph_evidence_text, 'figure'),
            Output(self.graph_evidence_vis, 'figure'),
            Output(self.visual_evidences, 'figure'),
            Output(self.claim_text, 'children'),
            Output(self.text_evidences, 'children'),
            Output(self.evidence_text, 'children'),
            Output(self.evidence_image_alt_text, 'children'),
            Output(self.predicted_label, 'children'),
            Output(self.actual_label, 'children'),
            Output(self.num_claim_edges, 'children'),
            Output(self.frac_verified, 'children'),
            Output(self.explanations, 'children'),
            Output(self.visual_similarity_score, 'children'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update)
