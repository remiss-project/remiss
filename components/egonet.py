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
                                      config={'displayModeBar': False},
                                      responsive=True,
                                      # style={'height': '100%', 'width': '100%'},
                                      )
        available_users = self.plot_factory.get_users(self.available_datasets[0])
        self.user_dropdown = dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users],
                                          # value=available_users[0],
                                          id=f'user-dropdown-{self.name}', disabled=True)
        self.depth_slider = dcc.Slider(min=1, max=5, step=1, value=2, id=f'slider-{self.name}')
        self.user_checkbox = dcc.Checklist(options=[{"label": " Show egonet", "value": 1}], value=[],
                                           id=f'user-checkbox-{self.name}')

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Filtered network', id=f'title-{self.name}'),
            dbc.CardBody([
                dcc.Loading(id=f'loading-{self.name}',
                            type='default',
                            children=self.graph_egonet)
            ]),
            dbc.CardFooter([
                dbc.Row([
                    dbc.Col([
                        self.user_checkbox
                    ], width=6),
                ]),
                dbc.Collapse([
                    dbc.Row([
                        dbc.Col([
                            self.user_dropdown
                        ], width=6),
                        dbc.Col([
                            self.depth_slider
                        ], width=6),
                    ], style={'padding': '0px 50px 50px 0px'})
                ], id=f'collapse-user-{self.name}', is_open=False),

                html.Div([], style={'height': '100%'}, id=f'placeholder-{self.name}') if self.debug else None,
            ])
        ], style={'height': '100%'})

    def update(self, dataset, egonet_user, user, start_date, end_date, hashtags, user_disabled, depth):
        logger.info(f'Updating egonet with dataset {dataset}, user {user}, egonet user {egonet_user}, '
                    f'start date {start_date}, end date {end_date}, hashtag {hashtags}, depth {depth}, '
                    f'user disabled {user_disabled}')
        if user_disabled:
            egonet_user = None

        if ctx.triggered_id == f'user-dropdown-{self.name}' or ctx.triggered_id == f'slider-{self.name}':
            # Show egonet for the selected user
            try:
                fig = self.plot_factory.plot_egonet(dataset, egonet_user, depth, start_date, end_date, hashtags)
                title = f'Egonet'
                logger.info(f'Plotting egonet for user {egonet_user}')
            except (RuntimeError, ValueError, KeyError) as e:
                # If the user is not available, then show the backbone
                fig = self.plot_factory.plot_hidden_network(dataset, start_date=start_date, end_date=end_date,
                                                            hashtag=hashtags)
                title = 'Filtered network'
                logger.info(f'User {egonet_user} not available, plotting backbone')
        else:
            # Plot backbone but highlight the selected user
            fig = self.plot_factory.plot_hidden_network(dataset, user, start_date, end_date, hashtags)
            title = 'Filtered network'
            logger.info(f'Plotting backbone with table user {user} highlighted')
        # remove margin
        # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig, title

    def update_user(self, user):
        logger.info(f'Updating egonet user with {user}')
        return user

    def update_current_date(self, date):
        logger.info(f'Updating egonet date with {date}')
        return date

    def update_user_checkbox(self, checkbox_value):
        return not checkbox_value, not checkbox_value

    def update_date_checkbox(self, checkbox_value):
        return not checkbox_value

    def update_user_collapse(self, checkbox_value):
        return bool(checkbox_value)

    def update_date_collapse(self, checkbox_value):
        return bool(checkbox_value)

    def update_user_list(self, dataset):
        available_users = self.plot_factory.get_users(dataset)
        available_users.columns = ['label', 'value']
        return available_users.to_dict('records')

    def update_debug(self, dataset, user, depth, user_checkbox,
                     user_dropdown_disabled, depth_slider_disabled):
        return f'Dataset: {dataset}, User: {user}, Depth: {depth}, ' \
               f'User checkbox: {user_checkbox},  ' \
               f'User dropdown disabled: {user_dropdown_disabled}, Depth slider disabled: {depth_slider_disabled}, '

    def callbacks(self, app):
        app.callback(
            Output(f'fig-{self.name}', 'figure'),
            Output(f'title-{self.name}', 'children'),
            [Input(self.state.current_dataset, 'data'),
             Input(f'user-dropdown-{self.name}', 'value'),
             Input(self.state.current_user, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(self.state.current_hashtags, 'data'),
             Input(self.user_dropdown, 'disabled'),
             Input(self.depth_slider, 'value')],
        )(self.update)
        app.callback(
            Output(self.user_dropdown, 'value'),
            [Input(self.state.current_user, 'data')],
        )(self.update_user)

        app.callback(
            Output(self.user_dropdown, 'options'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update_user_list)
        app.callback(
            Output(self.user_dropdown, 'disabled'),
            Output(self.depth_slider, 'disabled'),
            Input(self.user_checkbox, 'value'),
        )(self.update_user_checkbox)

        app.callback(
            Output(f'collapse-user-{self.name}', 'is_open'),
            Input(self.user_checkbox, 'value'),
        )(self.update_user_collapse)

        if self.debug:
            app.callback(
                Output(f'placeholder-{self.name}', 'children'),
                [Input(self.state.current_dataset, 'data'),
                 Input(self.state.current_user, 'data'),
                 Input(self.depth_slider, 'value'),
                 Input(self.user_checkbox, 'value'),
                 Input(self.user_dropdown, 'disabled'),
                 Input(self.depth_slider, 'disabled'),
                 ]
            )(self.update_debug)
