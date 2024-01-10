from abc import ABC

import dash_bootstrap_components as dbc
import pandas as pd
import shortuuid
from dash import dcc, html, Input, Output, State
from dash.dash_table import DataTable
from dash_holoniq_wordcloud import DashWordcloud


def patch_layout_ids(layout, name):
    try:
        layout.id = f'{layout.id}-{name}'
    except AttributeError:
        pass
    try:
        for child in layout.children:
            patch_layout_ids(child, name)
    except AttributeError:
        pass


class DashComponent(ABC):
    def __init__(self, name=None):
        self.name = name if name else str(shortuuid.ShortUUID().random(length=10))
        # f'{slugify(self.__class__.__name__)}-{shortuuid.ShortUUID().random(length=10)}'

    def layout(self, params=None):
        raise NotImplementedError()

    def callbacks(self, app):
        pass


class TweetUserTimeSeriesComponent(DashComponent):
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
                    dbc.CardHeader('Tweet frequency'),
                    dbc.CardBody([
                        self.graph_tweet
                    ])
                ]),
            ]),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('User frequency'),
                    dbc.CardBody([
                        self.graph_users
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, hashtags, start_date, end_date):
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


class TopTableComponent(DashComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.data = None
        self.state = state
        self.table = DataTable(data=[], id=f'table-{self.name}',
                               columns=[{"name": i, "id": i} for i in self.plot_factory.top_table_columns],
                               editable=False,
                               filter_action="native",
                               sort_action="native",
                               sort_mode="multi",
                               # column_selectable="multi",
                               # row_selectable="single",
                               row_deletable=False,
                               selected_columns=[],
                               selected_rows=[],
                               page_action="native",
                               page_current=0,
                               page_size=10,
                               style_cell={
                                   'overflow': 'hidden',
                                   'textOverflow': 'ellipsis',
                                   'maxWidth': 0,
                               },
                               style_cell_conditional=[
                                   {'if': {'column_id': 'Text'},
                                    'width': '60%'},
                               ]

                               )

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                self.table
            ], width=12),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date):
        self.data = self.plot_factory.get_top_table_data(dataset, start_date, end_date)
        return self.data.to_dict('records')

    def update_hashtags(self, active_cell):
        if active_cell:
            hashtags = self.extract_hashtag_from_top_table(active_cell)
            if hashtags:
                return hashtags
        return None

    def update_user(self, active_cell):
        if active_cell:
            user = self.data['User'].iloc[active_cell['row']]
            return user
        return None

    def extract_hashtag_from_top_table(self, active_cell):
        text = self.data['Text'].iloc[active_cell['row']]
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None

    def callbacks(self, app):
        app.callback(
            Output(self.table, 'data'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)
        app.callback(
            Output(self.state.current_hashtags, 'data', allow_duplicate=True),
            [Input(self.table, 'active_cell')],
        )(self.update_hashtags)
        app.callback(
            Output(self.state.current_user, 'data'),
            [Input(self.table, 'active_cell')],
        )(self.update_user)


class EgonetComponent(DashComponent):
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
                                      style={'height': '100%', 'width': '100%'},
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
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Egonet'),
                    dbc.CardBody([
                        self.graph_egonet
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
                        html.Div([], style={'height': '100%'},
                                 id=f'placeholder-{self.name}') if self.debug else None,

                    ])

                ], class_name='h-100')
            ], class_name='h-100'),
        ], justify='center', class_name='h-100 w-100', style={'margin-bottom': '1rem'})

    def update(self, dataset, user, depth, date_index, user_disabled, depth_disabled, date_disabled):
        if user_disabled:
            user = None
        if depth_disabled:
            depth = None
        if date_disabled:
            start_date = None
            end_date = None
        else:
            start_date = self.dates[date_index]
            end_date = self.dates[date_index + 1]

        fig = self.plot_factory.plot_egonet(dataset, user, depth, start_date, end_date)
        # remove margin
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
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
             Input(self.depth_slider, 'value'),
             Input(self.date_slider, 'value'),
             Input(self.user_dropdown, 'disabled'),
             Input(self.depth_slider, 'disabled'),
             Input(self.date_slider, 'disabled')],
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


class ControlPanelComponent(DashComponent):
    def __init__(self, plot_factory, state, name=None,
                 max_wordcloud_words=100, wordcloud_width=800, wordcloud_height=400, match_wordcloud_width=False):
        super().__init__(name)
        self.state = state
        self.wordcloud_height = wordcloud_height
        self.wordcloud_width = wordcloud_width
        self.match_wordcloud_width = match_wordcloud_width
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.max_wordcloud_words = max_wordcloud_words
        self.date_picker = self.get_date_picker_component()
        self.wordcloud = self.get_wordcloud_component()
        self.dataset_dropdown = self.get_dataset_dropdown_component()

    def get_wordcloud_component(self):
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(self.available_datasets[0])
        if self.max_wordcloud_words:
            print(f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
        min_freq = min([x[1] for x in available_hashtags_freqs])

        return DashWordcloud(
            list=available_hashtags_freqs,
            width=self.wordcloud_width, height=self.wordcloud_height,
            rotateRatio=0.5,
            shrinkToFit=True,
            weightFactor=10 / min_freq,
            hover=True,
            id=f'wordcloud-{self.name}'
            ,
        )

    def get_dataset_dropdown_component(self):
        return dcc.Dropdown(options=[{"label": x, "value": x} for x in self.available_datasets],
                            value=self.available_datasets[0],
                            id=f'dataset-dropdown-{self.name}')

    def get_date_picker_component(self):
        min_date_allowed, max_date_allowed, start_date, end_date = self.update_date_range(self.available_datasets[0])
        return dcc.DatePickerRange(
            id=f'date-picker-{self.name}',
            min_date_allowed=min_date_allowed,
            max_date_allowed=max_date_allowed,
            initial_visible_month=min_date_allowed,
            start_date=min_date_allowed,
            end_date=max_date_allowed)

    def update_wordcloud(self, dataset):
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(dataset)
        if self.max_wordcloud_words:
            print(f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
        min_freq = min([x[1] for x in available_hashtags_freqs])

        return available_hashtags_freqs, 10 / min_freq

    def update_date_range(self, dataset):
        min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(dataset)
        return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed

    def update_dataset_storage(self, dropdown_dataset):
        return dropdown_dataset

    def update_hashtag_storage(self, click_data):
        return [click_data[0][0]] if click_data and len(click_data) == 1 else None

    def update_start_date_storage(self, start_date):
        return start_date

    def update_end_date_storage(self, end_date):
        return end_date

    def layout(self, params=None):
        return dbc.Stack([
            dbc.Card([
                dbc.CardHeader('Dataset'),
                dbc.CardBody([
                    self.dataset_dropdown
                ])
            ]),

            dbc.Card([
                dbc.CardHeader('Date range'),
                dbc.CardBody([
                    self.date_picker
                ])
            ]),

            dbc.Card([
                dbc.CardHeader('Hashtags'),
                dbc.CardBody([
                    self.wordcloud
                ])
            ]),
        ], gap=2, style={'width': f'{self.wordcloud_width + 40}px'} if self.match_wordcloud_width else None)

    def callbacks(self, app):

        app.callback(
            Output(self.date_picker, 'min_date_allowed'),
            Output(self.date_picker, 'max_date_allowed'),
            Output(self.date_picker, 'start_date'),
            Output(self.date_picker, 'end_date'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update_date_range)

        app.callback(
            Output(self.state.current_dataset, 'data'),
            [Input(self.dataset_dropdown, 'value')],
        )(self.update_dataset_storage)

        app.callback(
            Output(self.state.current_hashtags, 'data'),
            Input(self.wordcloud, 'click'),
        )(self.update_hashtag_storage)

        app.callback(
            Output(self.state.current_start_date, 'data'),
            [Input(self.date_picker, 'start_date')],
        )(self.update_start_date_storage)

        app.callback(
            Output(self.state.current_end_date, 'data'),
            [Input(self.date_picker, 'end_date')],
        )(self.update_end_date_storage)


class RemissState(DashComponent):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.current_dataset = dcc.Store(id=f'current-dataset-{self.name}')
        self.current_hashtags = dcc.Store(id=f'current-hashtags-{self.name}')
        self.current_start_date = dcc.Store(id=f'current-start-date-{self.name}')
        self.current_end_date = dcc.Store(id=f'current-end-date-{self.name}')
        self.current_user = dcc.Store(id=f'current-user-{self.name}')

    def layout(self, params=None):
        return html.Div([
            self.current_dataset,
            self.current_hashtags,
            self.current_start_date,
            self.current_end_date,
            self.current_user,
        ])

    def callbacks(self, app):
        pass


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, top_table_factory, egonet_plot_factory, name=None,
                 max_wordcloud_words=100, wordcloud_width=400, wordcloud_height=400, match_wordcloud_width=True,

                 debug=False):
        super().__init__(name=name)
        self.debug = debug
        self.match_wordcloud_width = match_wordcloud_width
        self.wordcloud_height = wordcloud_height
        self.wordcloud_width = wordcloud_width
        self.max_wordcloud_words = max_wordcloud_words

        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.top_table_factory = top_table_factory
        self.available_datasets = tweet_user_plot_factory.available_datasets

        self.state = RemissState(name='state')

        self.top_table_component = TopTableComponent(top_table_factory,
                                                     state=self.state,
                                                     name='top')
        self.tweet_user_ts_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory,
                                                                    state=self.state,
                                                                    name='ts')

        self.egonet_component = EgonetComponent(egonet_plot_factory,
                                                state=self.state,
                                                name='egonet',
                                                debug=self.debug, )

        self.control_panel_component = ControlPanelComponent(tweet_user_plot_factory,
                                                             state=self.state,
                                                             name='control',
                                                             max_wordcloud_words=self.max_wordcloud_words,
                                                             wordcloud_width=self.wordcloud_width,
                                                             wordcloud_height=self.wordcloud_height,
                                                             match_wordcloud_width=self.match_wordcloud_width)

    def update_placeholder(self, dataset, hashtags, start_date, end_date, current_user):
        return html.H1(f'Hashtag: {hashtags}, Dataset: {dataset}, Start date: {start_date}, '
                       f'End date: {end_date}, Current user: {current_user}')

    def layout(self, params=None):
        return dbc.Container([
            self.state.layout(),
            dbc.NavbarSimple(
                brand="REMISS – Towards a methodology to reduce misinformation spread about vulnerable and stigmatised groups",
                brand_href="#",
                sticky="top",
                style={'font-size': '1.5rem', 'font-weight': 'bold', 'margin-bottom': '1rem'},
                fluid=True,

            ),
            html.Div([], style={'margin-bottom': '1rem'}, id=f'placeholder-{self.name}') if self.debug else None,
            dbc.Row([
                dbc.Col([
                    self.control_panel_component.layout(),
                ],
                    width='auto' if self.match_wordcloud_width else 4,
                    class_name='h-100',
                ),
                dbc.Col([
                    self.egonet_component.layout(),
                ],
                ),
            ], style={'margin-bottom': '1rem'}, justify='center'),
            self.top_table_component.layout(),
            self.tweet_user_ts_component.layout(),
        ], fluid=False)

    def callbacks(self, app):
        if self.debug:
            app.callback(
                Output(f'placeholder-{self.name}', 'children'),
                [Input(self.state.current_dataset, 'data'),
                 Input(self.state.current_hashtags, 'data'),
                 Input(self.state.current_start_date, 'data'),
                 Input(self.state.current_end_date, 'data'),
                 Input(self.state.current_user, 'data')],
            )(self.update_placeholder)
        self.control_panel_component.callbacks(app)
        self.tweet_user_ts_component.callbacks(app)
        self.top_table_component.callbacks(app)
        self.egonet_component.callbacks(app)
