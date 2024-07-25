import logging

import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, Input, Output, ctx

from components.components import RemissComponent

logger = logging.getLogger(__name__)


class EgonetComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None, debug=False):
        super().__init__(name=name)
        self.debug = debug
        self.dates = None
        self.plot_factory = plot_factory
        self.state = state
        self.available_datasets = plot_factory.available_datasets
        self.graph_egonet = dcc.Graph(figure={}, id=f'fig-{self.name}',
                                      # config={'displayModeBar': False},
                                      responsive=True,
                                      # style={'height': '100%', 'width': '100%'},
                                      )
        self.depth_slider = dcc.Slider(min=1, max=5, step=1, value=2, id=f'slider-{self.name}')

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Filtered network', id=f'title-{self.name}'),
            dbc.CardBody([
                dcc.Loading(id=f'loading-{self.name}',
                            type='default',
                            children=self.graph_egonet,
                            # style={'height': '100%'}
                            ),
            ]),
            dbc.Collapse([
                dbc.CardFooter([
                    dbc.Row([
                        dbc.Col([
                            self.depth_slider
                        ], width=6),
                    ]),
                ])
            ], id=f'collapse-depth-slider-{self.name}', is_open=False),
        ], style={'height': '100%'})

    def update(self, dataset, user, start_date, end_date, hashtags, depth):
        logger.info(f'Updating egonet with dataset {dataset}, user {user}, '
                    f'start date {start_date}, end date {end_date}, hashtag {hashtags}, depth {depth}')
        # Show egonet for the selected user
        try:
            fig = self.plot_factory.plot_egonet(dataset, user, depth, start_date, end_date, hashtags)
            title = f'Egonet'
            show_depth_slider = True
            logger.info(f'Plotting egonet for user {user}')
        except (RuntimeError, ValueError, KeyError) as e:
            # If the user is not available, then show the backbone
            fig = self.plot_factory.plot_hidden_network(dataset, start_date=start_date, end_date=end_date,
                                                        hashtag=hashtags)
            show_depth_slider = False
            title = 'Filtered network'
            logger.info(f'User {user} not available, plotting backbone')

        # remove margin
        # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig, title, show_depth_slider, not show_depth_slider

    def callbacks(self, app):
        app.callback(
            Output(f'fig-{self.name}', 'figure'),
            Output(f'title-{self.name}', 'children'),
            Output(f'collapse-depth-slider-{self.name}', 'is_open'),
            Output(self.depth_slider, 'disabled'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(self.state.current_hashtags, 'data'),
             Input(self.depth_slider, 'value'),
             ]
        )(self.update)
