import dash_bootstrap_components as dbc
from dash import dcc, Input, Output

from components.components import RemissComponent


class PropagationComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_tree = dcc.Graph(figure={}, id=f'fig-propagation-tree-{self.name}')
        self.graph_propagation_depth = dcc.Graph(figure={}, id=f'fig-propagation-depth-{self.name}')
        self.graph_propagation_size = dcc.Graph(figure={}, id=f'fig-propagation-size-{self.name}')
        self.graph_propagation_max_breadth = dcc.Graph(figure={}, id=f'fig-propagation-max-breadth-{self.name}')
        self.graph_propagation_structural_virality = dcc.Graph(figure={},
                                                               id=f'fig-propagation-structural-virality-{self.name}')

        self.graph_cascade_ccdf = dcc.Graph(figure={}, id=f'fig-cascade-ccdf-{self.name}')
        self.graph_cascade_count_over_time = dcc.Graph(figure={}, id=f'fig-cascade-count-over-time-{self.name}')
        self.state = state

    def layout(self, params=None):
        # Two rows:
        # 1. Propagation tree, depth, size, max breadth, structural virality
        # 2. Cascade CCDF, cascade count over time

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Tree'),
                        dbc.CardBody([
                            self.graph_propagation_tree
                        ])
                    ]),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Depth'),
                        dbc.CardBody([
                            self.graph_propagation_depth
                        ])
                    ]),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Size'),
                        dbc.CardBody([
                            self.graph_propagation_size
                        ])
                    ]),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Max Breadth'),
                        dbc.CardBody([
                            self.graph_propagation_max_breadth
                        ])
                    ]),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Structural Virality'),
                        dbc.CardBody([
                            self.graph_propagation_structural_virality
                        ])
                    ]),
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Cascade CCDF'),
                        dbc.CardBody([
                            self.graph_cascade_ccdf
                        ])
                    ]),
                ], width=6),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Cascade Count Over Time'),
                        dbc.CardBody([
                            self.graph_cascade_count_over_time
                        ])
                    ]),
                ], width=6),
            ]),
        ])

    def update_tweet(self, dataset, tweet_id):
        return self.plot_factory.plot_propagation_tree(dataset, tweet_id), \
            self.plot_factory.plot_depth(dataset, tweet_id), \
            self.plot_factory.plot_size(dataset, tweet_id), \
            self.plot_factory.plot_max_breadth(dataset, tweet_id), \
            self.plot_factory.plot_structural_virality(dataset, tweet_id)

    def update_cascade(self, dataset):
        return self.plot_factory.plot_size_cascade_ccdf(dataset), \
            self.plot_factory.plot_cascade_count_over_time(dataset)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_tree, 'figure'),
            Output(self.graph_propagation_depth, 'figure'),
            Output(self.graph_propagation_size, 'figure'),
            Output(self.graph_propagation_max_breadth, 'figure'),
            Output(self.graph_propagation_structural_virality, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)

        app.callback(
            Output(self.graph_cascade_ccdf, 'figure'),
            Output(self.graph_cascade_count_over_time, 'figure'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update_cascade)