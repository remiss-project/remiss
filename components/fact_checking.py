from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html
from components.components import RemissComponent


class FactCheckingComponent(RemissComponent):
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

    def callbacks(self, app):
        pass
