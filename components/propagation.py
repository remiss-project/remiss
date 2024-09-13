import logging
from datetime import datetime

import dash_bootstrap_components as dbc
from dash import dcc, Input, Output

from components.components import RemissComponent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PropagationComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None, gap=2):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_tree = dcc.Graph(figure={}, id=f'fig-propagation-tree-{self.name}')
        self.graph_propagation_depth = dcc.Graph(figure={}, id=f'fig-propagation-depth-{self.name}')
        self.graph_propagation_size = dcc.Graph(figure={}, id=f'fig-propagation-size-{self.name}')
        self.graph_propagation_max_breadth = dcc.Graph(figure={}, id=f'fig-propagation-max-breadth-{self.name}')
        self.graph_propagation_structural_virality = dcc.Graph(figure={},
                                                               id=f'fig-propagation-structural-virality-{self.name}')

        self.state = state
        self.gap = gap

    def layout(self, params=None):
        """
        |                Propagation Tree                           |
        | Propagation Depth      |  Propagation size                |
        | Propagation Max Breadth | Propagation Structural Virality |
        :param params:
        :return:
        """

        return dbc.Collapse([
            dbc.Stack([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Propagation Tree'),
                            dbc.CardBody([
                                self.graph_propagation_tree
                            ]),
                            dbc.CardFooter([
                                'Propagation tree showing the structure of the cascade.'
                            ])
                        ]),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Propagation Depth'),
                            dbc.CardBody([
                                self.graph_propagation_depth
                            ]),
                            dbc.CardFooter([
                                "The depth of a cascade describes the longest path from the initial node(s) to the farthest affected node. It measures how deep the cascade penetrates the network, reflecting the number of layers or levels the effect reaches. A cascade with greater depth indicates that the influence of the initial event has traveled through many intermediary steps, affecting nodes that are several degrees of separation away from the origin. "
                            ])
                        ]),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Propagation Size'),
                            dbc.CardBody([
                                self.graph_propagation_size
                            ]),
                            dbc.CardFooter([
                                'The size of a cascade refers to the total number of nodes that are affected by the cascade. This metric provides a measure of the overall impact of the cascade on the network, indicating how widespread the effect has been. For instance, in the context of a failure cascade, the size would represent the number of nodes that experience failure because of the initial event. '
                            ])
                        ]),
                    ], width=6),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Propagation Max Breadth'),
                            dbc.CardBody([
                                self.graph_propagation_max_breadth
                            ]),
                            dbc.CardFooter([
                                'The maximum breadth of a cascade is the largest number of nodes affected at any single level of the cascade. This metric highlights the peak level of simultaneous impact, showing where the cascade has the widest reach in terms of the number of nodes affected at once. Maximum breadth is crucial for understanding the points of maximum network stress or influence during the cascade. '
                            ])
                        ]),
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Propagation Structural Virality'),
                            dbc.CardBody([
                                self.graph_propagation_structural_virality
                            ]),
                            dbc.CardFooter([
                                'Structural virality is a measure that combines elements of both breadth and depth, providing a nuanced view of how the cascade spreads through the network. It captures the overall shape and complexity of the cascade, reflecting how far and how wide the cascade spreads. Structural virality considers both the extent of the spread (like depth) and the distribution of the spread across different nodes (like breadth). A cascade with high structural virality indicates a complex, tree-like spread with many branches, as opposed to a simple, linear spread. '
                            ])
                        ]),
                    ], width=6),
                ]),
            ], gap=self.gap),
        ], id=f'collapse-{self.name}', is_open=False)

    def update_tweet(self, dataset, tweet_id):
        try:
            if self.plot_factory.get_cascade_size(dataset, tweet_id) > 1:
                logger.debug(f'Updating propagation plots with dataset {dataset}, tweet {tweet_id}')
                start_time = datetime.now()
                prop_tree = self.plot_factory.plot_propagation_tree(dataset, tweet_id)
                depth =    self.plot_factory.plot_depth_over_time(dataset, tweet_id)
                size =     self.plot_factory.plot_size_over_time(dataset, tweet_id)
                max_breath =  self.plot_factory.plot_max_breadth_over_time(dataset, tweet_id)
                structural =  self.plot_factory.plot_structural_virality_over_time(dataset, tweet_id)
                show =  True
                logger.debug(f'Propagation plots updated in {datetime.now() - start_time}')
                return prop_tree, depth, size, max_breath, structural, show
            else:
                logger.debug(f'Tweet {tweet_id} has no propagation')
                return {}, {}, {}, {}, {}, False

        except Exception as e:
            logger.debug(e)
            return {}, {}, {}, {}, {}, False

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_tree, 'figure'),
            Output(self.graph_propagation_depth, 'figure'),
            Output(self.graph_propagation_size, 'figure'),
            Output(self.graph_propagation_max_breadth, 'figure'),
            Output(self.graph_propagation_structural_virality, 'figure'),
            Output(f'collapse-{self.name}', 'is_open'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)


class CascadeCcdfComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_cascade_ccdf = dcc.Graph(figure={}, id=f'fig-cascade-ccdf-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Cascade CCDF'),
            dbc.CardBody([
                dcc.Loading(id=f'loading-cascade-ccdf-{self.name}',
                            type='default',
                            children=self.graph_cascade_ccdf)
            ]),
            dbc.CardFooter('Cascade Cumulative Distribution Function showing the probability of a cascade '
                           'reaching a certain size over time.')

        ])

    def update_cascade(self, dataset):
        try:
            logger.debug(f'Updating cascade ccdf with dataset {dataset}')
            return self.plot_factory.plot_size_cascade_ccdf(dataset)
        except KeyError as e:
            logger.debug(e)
            return {}

    def callbacks(self, app):
        app.callback(
            Output(self.graph_cascade_ccdf, 'figure'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update_cascade)


class CascadeCountOverTimeComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_cascade_count_over_time = dcc.Graph(figure={}, id=f'fig-cascade-count-over-time-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Cascade Count Over Time'),
            dbc.CardBody([
                dcc.Loading(id=f'loading-cascade-count-over-time-{self.name}',
                            type='default',
                            children=self.graph_cascade_count_over_time)
            ]),
            dbc.CardFooter('Cascade count over time showing the number of cascades over time.')
        ])

    def update_cascade(self, dataset):
        logger.debug(f'Updating cascade count over time with dataset {dataset}')
        return self.plot_factory.plot_cascade_count_over_time(dataset)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_cascade_count_over_time, 'figure'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update_cascade)


class PropagationTreeComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_tree = dcc.Graph(figure={}, id=f'fig-propagation-tree-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Tree'),
                        dbc.CardBody([
                            dcc.Loading(id=f'loading-propagation-tree-{self.name}',
                                        type='default',
                                        children=self.graph_propagation_tree)
                        ])
                    ]),
                ]),
            ]),
        ])

    def update_tweet(self, dataset, tweet_id):
        return self.plot_factory.plot_propagation_tree(dataset, tweet_id)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_tree, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)


class PropagationDepthComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_depth = dcc.Graph(figure={}, id=f'fig-propagation-depth-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Depth'),
                        dbc.CardBody([
                            dcc.Loading(id=f'loading-propagation-depth-{self.name}',
                                        type='default',
                                        children=self.graph_propagation_depth)
                        ])
                    ]),
                ]),
            ]),
        ])

    def update_tweet(self, dataset, tweet_id):
        return self.plot_factory.plot_depth_over_time(dataset, tweet_id)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_depth, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)


class PropagationSizeComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_size = dcc.Graph(figure={}, id=f'fig-propagation-size-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Size'),
                        dbc.CardBody([
                            dcc.Loading(id=f'loading-propagation-size-{self.name}',
                                        type='default',
                                        children=self.graph_propagation_size)
                        ])
                    ]),
                ]),
            ]),
        ])

    def update_tweet(self, dataset, tweet_id):
        return self.plot_factory.plot_size_over_time(dataset, tweet_id)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_size, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)


class PropagationMaxBreadthComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_max_breadth = dcc.Graph(figure={}, id=f'fig-propagation-max-breadth-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Max Breadth'),
                        dbc.CardBody([
                            dcc.Loading(id=f'loading-propagation-max-breadth-{self.name}',
                                        type='default',
                                        children=self.graph_propagation_max_breadth)

                        ])
                    ]),
                ]),
            ]),
        ])

    def update_tweet(self, dataset, tweet_id):
        return self.plot_factory.plot_max_breadth_over_time(dataset, tweet_id)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_max_breadth, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)


class PropagationStructuralViralityComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_propagation_structural_virality = dcc.Graph(figure={},
                                                               id=f'fig-propagation-structural-virality-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader('Propagation Structural Virality'),
                        dbc.CardBody([
                            dcc.Loading(id=f'loading-propagation-structural-virality-{self.name}',
                                        type='default',
                                        children=self.graph_propagation_structural_virality)
                        ])
                    ]),
                ]),
            ]),
        ])

    def update_tweet(self, dataset, tweet_id):
        return self.plot_factory.plot_structural_virality_over_time(dataset, tweet_id)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_propagation_structural_virality, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_tweet, 'data')],
        )(self.update_tweet)
