import logging

import dash_bootstrap_components as dbc
from dash import dcc, Output, Input

from .components import RemissComponent

logger = logging.getLogger(__name__)


class EmotionPerHourComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Evolution of Emotional Levels in Messages Across the Day', style={'fontSize': '18px', 'fontWeight': 'bold'}),
            dbc.CardBody([
                dcc.Loading(id=f'loading-{self.name}',
                            type='default',
                            children=self.graph)
            ]),
            dbc.CardFooter('This plot showcases how different emotions vary in intensity and frequency throughout the day, as analysed using the ROBERTUITO model. By mapping emotional trends across 24 hours, this visualization reveals patterns in emotional expression linked to specific times, capturing shifts in user sentiment from morning to night. This time-based analysis provides insights into how daily routines and events influence public sentiment on social networks, reflecting the dynamic nature of online interactions.')
        ])

    def update(self, dataset):
        try:
            logger.debug(
                f'Updating emotion per hour with dataset {dataset}')
            return self.plot_factory.plot_emotion_per_hour(dataset)
        except (IndexError, ValueError) as e:
            logger.error(f'Error updating emotion per hour: {e}')
            return {}

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data')],

        )(self.update)


class AverageEmotionBarComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Distribution of emotional levels in the messages', style={'fontSize': '18px', 'fontWeight': 'bold'}),
            dbc.CardBody([
                dcc.Loading(id=f'loading-{self.name}',
                            type='default',
                            children=self.graph)
            ]),
            dbc.CardFooter('This plot displays the intensity and frequency of various emotions detected in Twitter messages, as analyzed using the ROBERTUITO model. Each emotion category is represented by distinct levels, providing insight into the predominant emotional trends and sentiment fluctuations across the dataset.')
        ])

    def update(self, dataset):
        try:
            logger.debug(
                f'Updating average emotion with dataset {dataset}')
            return self.plot_factory.plot_average_emotion(dataset)
        except (IndexError, ValueError) as e:
            logger.error(f'Error updating average emotion: {e}')
            return {}

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update)


class TopProfilesComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Top profiles', style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        self.graph
                    ]),
                    dbc.CardFooter('Top profiles by number of tweets')
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        return self.plot_factory.plot_top_profiles(dataset, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)


class TopHashtagsComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Top hashtags', style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        self.graph
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        return self.plot_factory.plot_top_hashtags(dataset, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)


class TopicRankingComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Topic ranking', style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        self.graph
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        return self.plot_factory.plot_topic_ranking(dataset, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)


class NetworkTopicsComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Network topics', style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        self.graph
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        return self.plot_factory.plot_network_topics(dataset, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)
