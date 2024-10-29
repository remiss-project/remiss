import logging

import dash_bootstrap_components as dbc
from dash import dcc, Input, Output

from components.components import RemissComponent

logger = logging.getLogger(__name__)

class TimeSeriesComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_tweet = dcc.Graph(figure={}, id=f'fig-tweet-{self.name}')
        self.graph_users = dcc.Graph(figure={}, id=f'fig-users-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Distribution of Tweets Over Time',
                                   style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        dcc.Loading(id=f'loading-tweet-{self.name}',
                                    type='default',
                                    children=self.graph_tweet)
                    ]),
                    dbc.CardFooter('This figure illustrates the volume of tweets posted at different times, highlighting peak activity days and fluctuations. This visualisation offers insights into user behavior patterns, revealing when conversations are most active and potentially influenced by daily routines or global events.')
                ]),
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Distribution of Users Over Time',
                                   style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        dcc.Loading(id=f'loading-users-{self.name}',
                                    type='default',
                                    children=self.graph_users)
                    ]),
                    dbc.CardFooter('This figure illustrated the user activity throughout the time, identifying peaks when the highest numbers of users are active. By mapping user engagement patterns, this visualisation provides insights into temporal cycles of user presence and highlights key dates of interaction.')
                ]),
            ]),
        ])

    def update(self, dataset, hashtags, start_date, end_date):
        logger.debug(f'Updating time series with dataset {dataset}, hashtags {hashtags}, '
                    f'start date {start_date}, end date {end_date}')
        return self.plot_factory.plot_tweet_series(dataset, hashtags, start_date, end_date), \
            self.plot_factory.plot_user_series(dataset, hashtags, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_tweet, 'figure'),
            Output(self.graph_users, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_hashtags, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)
