import logging
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input
from components.components import RemissComponent

logger = logging.getLogger(__name__)


class MultimodalComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None, gap=2):
        super().__init__(name=name)
        self.gap = gap
        self.claim_text = html.P(id=f'{self.name}-claim_text')
        self.claim_image = dcc.Graph(figure={}, id=f'fig-{self.name}-claim_image')
        self.visual_evidence_text = html.P(id=f'{self.name}-visual_evidence_text')
        self.evidence_image = dcc.Graph(figure={}, id=f'fig-{self.name}-evidence_image')
        self.text_evidence_similarity_score = html.P(id=f'{self.name}-text_evidence_similarity_score')
        self.visual_evidence_similarity_score = html.P(id=f'{self.name}-visual_evidence_similarity_score')
        self.visual_evidence_graph_similarity_score = html.P(id=f'{self.name}-visual_evidence_graph_similarity_score')
        self.visual_evidence_domain = html.P(id=f'{self.name}-visual_evidence_domain')

        self.plot_factory = plot_factory
        self.state = state

    def layout(self, params=None):
        claim_column = dbc.Col(
            dbc.Stack([
                dbc.Card([
                    dbc.CardHeader('Claim Text'),
                    dbc.CardBody([
                        self.claim_text,
                    ])
                ]),
                dbc.Card([
                    dbc.CardHeader('Claim Image'),
                    dbc.CardBody([
                        self.claim_image,
                    ])
                ]),
            ], gap=self.gap),
            width=4)

        evidence_column = dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Stack([
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Text'),
                                    dbc.CardBody([
                                        self.visual_evidence_text,
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Evidence Image'),
                                    dbc.CardBody([
                                        self.evidence_image,
                                    ])
                                ]),
                            ], gap=self.gap)
                        ], width=6),
                        dbc.Col(
                            dbc.Stack([
                                dbc.Card([
                                    dbc.CardHeader('Text Evidence Similarity Score'),
                                    dbc.CardBody([
                                        self.text_evidence_similarity_score
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Similarity Score'),
                                    dbc.CardBody([
                                        self.visual_evidence_similarity_score
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Graph Similarity Score'),
                                    dbc.CardBody([
                                        self.visual_evidence_graph_similarity_score
                                    ])
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Domain'),
                                    dbc.CardBody([
                                        self.visual_evidence_domain
                                    ])
                                ]),
                            ], gap=self.gap),
                            width=6),
                    ])
                ])
            ])
        ], width=8)

        return dbc.Collapse([
            dbc.Row([
                claim_column,
                evidence_column,
            ]),

        ], id=f'{self.name}-collapse', is_open=False)

    def update(self, dataset, tweet_id):
        try:
            metadata = self.plot_factory.load_data_for_tweet(dataset, tweet_id)
            claim_text = metadata['claim_text']
            visual_evidence_text = metadata['visual_evidence_text']
            text_evidence_similarity_score = metadata['text_evidence_similarity_score']
            visual_evidence_similarity_score = metadata['visual_evidence_similarity_score']
            visual_evidence_graph_similarity_score = metadata['visual_evidence_graph_similarity_score']
            visual_evidence_domain = metadata['visual_evidence_domain']

            text_evidence_similarity_score = f'{float(text_evidence_similarity_score):.3f}'
            visual_evidence_similarity_score = f'{float(visual_evidence_similarity_score):.3f}'
            visual_evidence_graph_similarity_score = f'{float(visual_evidence_graph_similarity_score):.3f}'

            claim_image = self.plot_factory.plot_claim_image(dataset, tweet_id)
            evidence_image = self.plot_factory.plot_evidence_image(dataset, tweet_id)

            is_open = True
            return (claim_text, visual_evidence_text, text_evidence_similarity_score, visual_evidence_similarity_score,
                    visual_evidence_graph_similarity_score, visual_evidence_domain, claim_image, evidence_image,
                    is_open)
        except Exception as e:
            logger.error(f'Error updating multimodal component: {e}')
            return None, None, None, None, None, None, None, None, False

    def callbacks(self, app):
        app.callback(
            [Output(f'{self.name}-claim_text', 'children'),
             Output(f'{self.name}-visual_evidence_text', 'children'),
             Output(f'{self.name}-text_evidence_similarity_score', 'children'),
             Output(f'{self.name}-visual_evidence_similarity_score', 'children'),
             Output(f'{self.name}-visual_evidence_graph_similarity_score', 'children'),
             Output(f'{self.name}-visual_evidence_domain', 'children'),
             Output(f'fig-{self.name}-claim_image', 'figure'),
             Output(f'fig-{self.name}-evidence_image', 'figure'),
             Output(f'{self.name}-collapse', 'is_open')],
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update)
