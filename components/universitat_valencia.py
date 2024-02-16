from dash import dcc, Output, Input
import dash_bootstrap_components as dbc

from .components import RemissComponent


class EmotionPerHourComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Emotion per hour'),
                    dbc.CardBody([
                        self.graph
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        return self.plot_factory.plot_emotion_line_per_hour(dataset, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)


class AverageEmotionBarComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Average emotion'),
                    dbc.CardBody([
                        self.graph
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        return self.plot_factory.plot_average_emotion_bar(dataset, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
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
                    dbc.CardHeader('Top profiles'),
                    dbc.CardBody([
                        self.graph
                    ])
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
                    dbc.CardHeader('Top hashtags'),
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
                    dbc.CardHeader('Topic ranking'),
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
                    dbc.CardHeader('Network topics'),
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
