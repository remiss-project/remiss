import logging

import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input
from dash.exceptions import PreventUpdate

from components.components import RemissComponent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MultimodalComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None, gap=2):
        super().__init__(name=name)
        self.gap = gap
        self.claim_text = html.P(id=f'{self.name}-claim_text')
        self.claim_image = dcc.Graph(figure={}, id=f'{self.name}-claim_image')
        self.visual_evidence_text = html.P(id=f'{self.name}-visual_evidence_text')
        self.evidence_image = dcc.Graph(figure={}, id=f'{self.name}-evidence_image')
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
                    ]),
                    dbc.CardFooter('Claim text extracted from the tweet')
                ]),
                dbc.Card([
                    dbc.CardHeader('Claim Image'),
                    dbc.CardBody([
                        dcc.Loading(id=f'{self.name}-claim_image_loading', children=[
                            self.claim_image], type='default')
                    ]),
                    dbc.CardFooter('Claim image extracted from the tweet')
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
                                    ]),
                                    dbc.CardFooter('Visual evidence text extracted from the image')

                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Evidence Image'),
                                    dbc.CardBody([
                                        dcc.Loading(id=f'{self.name}-evidence_image_loading', children=[
                                            self.evidence_image], type='default')

                                    ]),
                                    dbc.CardFooter('Visual evidence image extracted from the tweet')
                                ]),
                            ], gap=self.gap)
                        ], width=6),
                        dbc.Col(
                            dbc.Stack([
                                dbc.Card([
                                    dbc.CardHeader('Text Evidence Similarity Score'),
                                    dbc.CardBody([
                                        self.text_evidence_similarity_score
                                    ]),
                                    dbc.CardFooter('Similarity score between the claim and the visual evidence')
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Similarity Score'),
                                    dbc.CardBody([
                                        self.visual_evidence_similarity_score
                                    ]),
                                    dbc.CardFooter('Similarity score between the claim and the visual evidence')
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Graph Similarity Score'),
                                    dbc.CardBody([
                                        self.visual_evidence_graph_similarity_score
                                    ]),
                                    dbc.CardFooter('Similarity score between the claim and the visual evidence graph')
                                ]),
                                dbc.Card([
                                    dbc.CardHeader('Visual Evidence Domain'),
                                    dbc.CardBody([
                                        self.visual_evidence_domain
                                    ]),
                                    dbc.CardFooter('Source of the visual evidence')
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

            logger.debug(f'Updating multimodal component for tweet {tweet_id}')

            return (claim_image, evidence_image,
                    claim_text, visual_evidence_text, text_evidence_similarity_score, visual_evidence_similarity_score,
                    visual_evidence_graph_similarity_score, visual_evidence_domain,
                    )
        except Exception as e:
            if 'not found in dataset' in str(e):
                logger.debug(f'No multimodal data found for tweet {tweet_id}')

            else:
                logger.error(f'Error updating multimodal component: {e}')
            raise PreventUpdate()

    def update_collapse(self, dataset, tweet_id):
        return self.plot_factory.has_multimodal_data(dataset, tweet_id)

    def callbacks(self, app):
        app.callback(
            [Output(self.claim_image, 'figure'),
             Output(self.evidence_image, 'figure'),
             Output(self.claim_text, 'children'),
             Output(self.visual_evidence_text, 'children'),
             Output(self.text_evidence_similarity_score, 'children'),
             Output(self.visual_evidence_similarity_score, 'children'),
             Output(self.visual_evidence_graph_similarity_score, 'children'),
             Output(self.visual_evidence_domain, 'children'),],
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update)

        app.callback(
            Output(f'{self.name}-collapse', 'is_open'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_collapse)

