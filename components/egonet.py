import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html, Input, Output, ctx

from components.components import RemissComponent


class EgonetComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None, debug=False):
        super().__init__(name=name)
        self.debug = debug
        self.dates = None
        self.frequency = plot_factory.frequency
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
        self.date_slider = dcc.Slider(min=0, max=1, step=1, id=f'date-slider-{self.name}', included=False)
        self.user_checkbox = dcc.Checklist(options=[{"label": " Filter by user", "value": 1}], value=[],
                                           id=f'user-checkbox-{self.name}')
        self.date_checkbox = dcc.Checklist(options=[{"label": " Filter by date", "value": 1}], value=[],
                                           id=f'date-checkbox-{self.name}')

    def layout(self, params=None):
        return dbc.Card([
            dbc.CardHeader('Egonet'),
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

                dbc.Row([
                    dbc.Col([
                        self.date_checkbox
                    ], width=6),
                ]),
                dbc.Collapse([
                    dbc.Row([
                        dbc.Col([
                            self.date_slider
                        ], width=12),
                    ], style={'padding': '0px 50px 50px 0px'})
                ], id=f'collapse-date-{self.name}', is_open=False),
                html.Div([], style={'height': '100%'}, id=f'placeholder-{self.name}') if self.debug else None,
            ])
        ])

    def update(self, dataset, egonet_user, user, state_start_date, state_end_date, hashtag, depth, date_index, user_disabled,
               depth_disabled, date_disabled):
        if user_disabled:
            user = None
        if depth_disabled:
            depth = None

        if date_disabled:
            # if the date slider is disabled then pick the one from the main date picker
            start_date = state_start_date
            end_date = state_end_date
        else:
            start_date = self.dates[date_index]
            end_date = self.dates[date_index + 1]

        if ctx.triggered_id == f'user-dropdown-{self.name}':
            # Show egonet for the selected user
            try:
                user = user[0]
                fig = self.plot_factory.plot_egonet(dataset, egonet_user, depth, start_date, end_date, hashtag)
            except (RuntimeError, ValueError, KeyError) as e:
                # If the user is not available, then show the backbone
                fig = self.plot_factory.plot_hidden_network(dataset, depth, start_date, end_date, hashtag)
        else:
            # Plot backbone but highlight the selected user
            fig = self.plot_factory.plot_hidden_network(dataset, user, depth, start_date, end_date, hashtag)
        # remove margin
        # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig

    def update_user(self, user):
        return user

    def update_current_date(self, date):
        return date

    def update_date_slider(self, start_date, end_date):
        days = pd.date_range(start_date, end_date, freq=self.frequency)
        style = {'transform': 'rotate(45deg)', "white-space": "nowrap", 'text-align': 'center', 'font-size': '12px',
                 'margin-top': '1rem'}
        marks = {i: {'label': str(days[i].date()), 'style': style} for i in
                 range(0, len(days))}
        self.dates = days

        return 0, len(days) - 1, 0, marks

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
        return [{"label": x, "value": x} for x in available_users]

    def update_debug(self, dataset, user, depth, date_index, user_checkbox, date_checkbox,
                     user_dropdown_disabled, depth_slider_disabled, date_slider_disabled):
        return f'Dataset: {dataset}, User: {user}, Depth: {depth}, Date index: {date_index}, ' \
               f'User checkbox: {user_checkbox}, Date checkbox: {date_checkbox}, ' \
               f'User dropdown disabled: {user_dropdown_disabled}, Depth slider disabled: {depth_slider_disabled}, ' \
               f'Date slider disabled: {date_slider_disabled}'

    def callbacks(self, app):
        app.callback(
            Output(self.graph_egonet, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data'),
             Input(self.user_dropdown, 'value'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(self.state.current_hashtags, 'data'),
             Input(self.depth_slider, 'value'),
             Input(self.date_slider, 'value'),
             Input(self.user_dropdown, 'disabled'),
             Input(self.depth_slider, 'disabled'),
             Input(self.date_slider, 'disabled'),
             ],
        )(self.update)
        app.callback(
            Output(self.state.current_user, 'data', allow_duplicate=True),
            [Input(self.user_dropdown, 'value')],
        )(self.update_user)
        app.callback(
            Output(self.user_dropdown, 'value'),
            [Input(self.state.current_user, 'data')],
        )(self.update_user)
        app.callback(
            Output(self.date_slider, 'min'),
            Output(self.date_slider, 'max'),
            Output(self.date_slider, 'value'),
            Output(self.date_slider, 'marks'),
            [Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update_date_slider)
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
            Output(self.date_slider, 'disabled'),
            Input(self.date_checkbox, 'value'),
        )(self.update_date_checkbox)
        app.callback(
            Output(f'collapse-user-{self.name}', 'is_open'),
            Input(self.user_checkbox, 'value'),
        )(self.update_user_collapse)
        app.callback(
            Output(f'collapse-date-{self.name}', 'is_open'),
            Input(self.date_checkbox, 'value'),
        )(self.update_date_collapse)

        if self.debug:
            app.callback(
                Output(f'placeholder-{self.name}', 'children'),
                [Input(self.state.current_dataset, 'data'),
                 Input(self.state.current_user, 'data'),
                 Input(self.depth_slider, 'value'),
                 Input(self.date_slider, 'value'),
                 Input(self.user_checkbox, 'value'),
                 Input(self.date_checkbox, 'value'),
                 Input(self.user_dropdown, 'disabled'),
                 Input(self.depth_slider, 'disabled'),
                 Input(self.date_slider, 'disabled'),
                 ]
            )(self.update_debug)
