import logging
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html, Output, Input
from components.components import RemissComponent

logger = logging.getLogger(__name__)


class MultimodalComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.visual_evidence_domain = html.P(id=f'{self.name}-visual_evidence_domain')
        self.visual_evidence_matched_categories = html.P(id=f'{self.name}-visual_evidence_matched_categories')
        self.visual_evidence_text = html.P(id=f'{self.name}-visual_evidence_text')
        self.visual_evidence_similarity_score = html.P(id=f'{self.name}-visual_evidence_similarity_score')
        self.visual_evidence_graph_similarity_score = html.P(id=f'{self.name}-visual_evidence_graph_similarity_score')
        self.text_evidence = html.P(id=f'{self.name}-text_evidence')
        self.text_evidence_similarity_score = html.P(id=f'{self.name}-text_evidence_similarity_score')
        self.text_evidence_graph_similarity_score = html.P(id=f'{self.name}-text_evidence_graph_similarity_score')
        self.visual_evidence_domain1 = html.P(id=f'{self.name}-visual_evidence_domain1')
        self.visual_evidence_matched_categories1 = html.P(id=f'{self.name}-visual_evidence_matched_categories1')
        self.visual_evidence_text1 = html.P(id=f'{self.name}-visual_evidence_text1')
        self.visual_evidence_similarity_score1 = html.P(id=f'{self.name}-visual_evidence_similarity_score1')
        self.visual_evidence_graph_similarity_score1 = html.P(id=f'{self.name}-visual_evidence_graph_similarity_score1')
        self.claim_text = html.P(id=f'{self.name}-claim_text')
        self.found_flag = html.P(id=f'{self.name}-found_flag')
        self.t_sug = html.P(id=f'{self.name}-t_sug')
        self.old_t_sug = html.P(id=f'{self.name}-old_t_sug')

        self.graph_claim = dcc.Graph(figure={}, id=f'fig-{self.name}-graph_claim')
        self.graph_evidence_vis = dcc.Graph(figure={}, id=f'fig-{self.name}-graph_evidence_vis')
        self.graph_evidence_text = dcc.Graph(figure={}, id=f'fig-{self.name}-graph_evidence_text')
        self.evidence_image = dcc.Graph(figure={}, id=f'fig-{self.name}-evidence_image')
        self.graph_claim1 = dcc.Graph(figure={}, id=f'fig-{self.name}-graph_claim1')
        self.graph_evidence_vis1 = dcc.Graph(figure={}, id=f'fig-{self.name}-graph_evidence_vis1')
        self.graph_evidence_text1 = dcc.Graph(figure={}, id=f'fig-{self.name}-graph_evidence_text1')
        self.evidence_image1 = dcc.Graph(figure={}, id=f'fig-{self.name}-evidence_image1')
        self.claim_image = dcc.Graph(figure={}, id=f'fig-{self.name}-claim_image')

        self.plot_factory = plot_factory
        self.state = state

    def layout(self, params=None):
        # Display the multimodal data in a grid of cards of 3 columns per row
        return dbc.Collapse([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Domain'),
                        dbc.CardBody([
                            self.visual_evidence_domain
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Matched Categories'),
                        dbc.CardBody([
                            self.visual_evidence_matched_categories
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Text'),
                        dbc.CardBody([
                            self.visual_evidence_text
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Similarity Score'),
                        dbc.CardBody([
                            self.visual_evidence_similarity_score
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Graph Similarity Score'),
                        dbc.CardBody([
                            self.visual_evidence_graph_similarity_score
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Text Evidence'),
                        dbc.CardBody([
                            self.text_evidence
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Text Evidence Similarity Score'),
                        dbc.CardBody([
                            self.text_evidence_similarity_score
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Text Evidence Graph Similarity Score'),
                        dbc.CardBody([
                            self.text_evidence_graph_similarity_score
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Domain1'),
                        dbc.CardBody([
                            self.visual_evidence_domain1
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Matched Categories1'),
                        dbc.CardBody([
                            self.visual_evidence_matched_categories1
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Text1'),
                        dbc.CardBody([
                            self.visual_evidence_text1
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Similarity Score1'),
                        dbc.CardBody([
                            self.visual_evidence_similarity_score1
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Visual Evidence Graph Similarity Score1'),
                        dbc.CardBody([
                            self.visual_evidence_graph_similarity_score1
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Claim Text'),
                        dbc.CardBody([
                            self.claim_text
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Found Flag'),
                        dbc.CardBody([
                            self.found_flag
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('T Sug'),
                        dbc.CardBody([
                            self.t_sug
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Old T Sug'),
                        dbc.CardBody([
                            self.old_t_sug
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Graph Claim'),
                        dbc.CardBody([
                            self.graph_claim
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Graph Evidence Vis'),
                        dbc.CardBody([
                            self.graph_evidence_vis
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Graph Evidence Text'),
                        dbc.CardBody([
                            self.graph_evidence_text
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Evidence Image'),
                        dbc.CardBody([
                            self.evidence_image
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Graph Claim1'),
                        dbc.CardBody([
                            self.graph_claim1
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Graph Evidence Vis1'),
                        dbc.CardBody([
                            self.graph_evidence_vis1
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Graph Evidence Text1'),
                        dbc.CardBody([
                            self.graph_evidence_text1
                        ])
                    ]),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Evidence Image1'),
                        dbc.CardBody([
                            self.evidence_image1
                        ])
                    ]),
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Claim Image'),
                        dbc.CardBody([
                            self.claim_image
                        ])
                    ]),
                ]),
            ]),
        ], id=f'{self.name}-collapse', is_open=False)

    def update(self, dataset, tweet_id):
        try:
            metadata = self.plot_factory.load_data_for_tweet(dataset, tweet_id)
            visual_evidence_domain = metadata['visual_evidence_domain']
            visual_evidence_matched_categories = metadata['visual_evidence_matched_categories']
            visual_evidence_text = metadata['visual_evidence_text']
            visual_evidence_similarity_score = metadata['visual_evidence_similarity_score']
            visual_evidence_graph_similarity_score = metadata['visual_evidence_graph_similarity_score']
            text_evidence = metadata['text_evidence']
            text_evidence_similarity_score = metadata['text_evidence_similarity_score']
            text_evidence_graph_similarity_score = metadata['text_evidence_graph_similarity_score']
            visual_evidence_domain1 = metadata['visual_evidence_domain1']
            visual_evidence_matched_categories1 = metadata['visual_evidence_matched_categories1']
            visual_evidence_text1 = metadata['visual_evidence_text1']
            visual_evidence_similarity_score1 = metadata['visual_evidence_similarity_score1']
            visual_evidence_graph_similarity_score1 = metadata['visual_evidence_graph_similarity_score1']
            claim_text = metadata['claim_text']
            found_flag = metadata['found_flag']
            t_sug = metadata['t_sug']
            old_t_sug = metadata['old_t_sug']
            graph_claim = self.plot_factory.plot_graph_claim(dataset, tweet_id)
            graph_evidence_vis = self.plot_factory.plot_graph_evidence_vis(dataset, tweet_id)
            graph_evidence_text = self.plot_factory.plot_graph_evidence_text(dataset, tweet_id)
            evidence_image = self.plot_factory.plot_evidence_image(dataset, tweet_id)
            graph_claim1 = self.plot_factory.plot_graph_claim1(dataset, tweet_id)
            graph_evidence_vis1 = self.plot_factory.plot_graph_evidence_vis1(dataset, tweet_id)
            graph_evidence_text1 = self.plot_factory.plot_graph_evidence_text1(dataset, tweet_id)
            evidence_image1 = self.plot_factory.plot_evidence_image1(dataset, tweet_id)
            claim_image = self.plot_factory.plot_claim_image(dataset, tweet_id)
            is_open = True
            return (visual_evidence_domain, visual_evidence_matched_categories, visual_evidence_text,
                    visual_evidence_similarity_score, visual_evidence_graph_similarity_score, text_evidence,
                    text_evidence_similarity_score, text_evidence_graph_similarity_score, visual_evidence_domain1,
                    visual_evidence_matched_categories1, visual_evidence_text1, visual_evidence_similarity_score1,
                    visual_evidence_graph_similarity_score1, claim_text, found_flag, t_sug, old_t_sug, graph_claim,
                    graph_evidence_vis, graph_evidence_text, evidence_image, graph_claim1, graph_evidence_vis1,
                    graph_evidence_text1, evidence_image1, claim_image, is_open)
        except Exception as e:
            if tweet_id is not None:
                logger.error(f'Error updating multimodal: {e}')
            visual_evidence_domain = ''
            visual_evidence_matched_categories = ''
            visual_evidence_text = ''
            visual_evidence_similarity_score = ''
            visual_evidence_graph_similarity_score = ''
            text_evidence = ''
            text_evidence_similarity_score = ''
            text_evidence_graph_similarity_score = ''
            visual_evidence_domain1 = ''
            visual_evidence_matched_categories1 = ''
            visual_evidence_text1 = ''
            visual_evidence_similarity_score1 = ''
            visual_evidence_graph_similarity_score1 = ''
            claim_text = ''
            found_flag = ''
            t_sug = ''
            old_t_sug = ''
            graph_claim = {}
            graph_evidence_vis = {}
            graph_evidence_text = {}
            evidence_image = {}
            graph_claim1 = {}
            graph_evidence_vis1 = {}
            graph_evidence_text1 = {}
            evidence_image1 = {}
            claim_image = {}
            is_open = False
            return (visual_evidence_domain, visual_evidence_matched_categories, visual_evidence_text,
                    visual_evidence_similarity_score, visual_evidence_graph_similarity_score, text_evidence,
                    text_evidence_similarity_score, text_evidence_graph_similarity_score, visual_evidence_domain1,
                    visual_evidence_matched_categories1, visual_evidence_text1, visual_evidence_similarity_score1,
                    visual_evidence_graph_similarity_score1, claim_text, found_flag, t_sug, old_t_sug, graph_claim,
                    graph_evidence_vis, graph_evidence_text, evidence_image, graph_claim1, graph_evidence_vis1,
                    graph_evidence_text1, evidence_image1, claim_image, is_open)

    def callbacks(self, app):
        app.callback(
            Output(self.visual_evidence_domain, 'children'),
            Output(self.visual_evidence_matched_categories, 'children'),
            Output(self.visual_evidence_text, 'children'),
            Output(self.visual_evidence_similarity_score, 'children'),
            Output(self.visual_evidence_graph_similarity_score, 'children'),
            Output(self.text_evidence, 'children'),
            Output(self.text_evidence_similarity_score, 'children'),
            Output(self.text_evidence_graph_similarity_score, 'children'),
            Output(self.visual_evidence_domain1, 'children'),
            Output(self.visual_evidence_matched_categories1, 'children'),
            Output(self.visual_evidence_text1, 'children'),
            Output(self.visual_evidence_similarity_score1, 'children'),
            Output(self.visual_evidence_graph_similarity_score1, 'children'),
            Output(self.claim_text, 'children'),
            Output(self.found_flag, 'children'),
            Output(self.t_sug, 'children'),
            Output(self.old_t_sug, 'children'),
            Output(self.graph_claim, 'figure'),
            Output(self.graph_evidence_vis, 'figure'),
            Output(self.graph_evidence_text, 'figure'),
            Output(self.evidence_image, 'figure'),
            Output(self.graph_claim1, 'figure'),
            Output(self.graph_evidence_vis1, 'figure'),
            Output(self.graph_evidence_text1, 'figure'),
            Output(self.evidence_image1, 'figure'),
            Output(self.claim_image, 'figure'),
            Output(f'{self.name}-collapse', 'is_open'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')]
        )(self.update)
