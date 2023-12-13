from abc import ABC

import dash_bootstrap_components as dbc
import shortuuid
from dash import dcc, html, Input, Output, callback_context
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
    def __init__(self, plot_factory, current_dataset, current_hashtags, current_start_date, current_end_date,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph_tweet = dcc.Graph(figure={}, id=f'fig-tweet-{self.name}')
        self.graph_users = dcc.Graph(figure={}, id=f'fig-users-{self.name}')
        self.current_dataset = current_dataset
        self.current_hashtags = current_hashtags
        self.current_start_date = current_start_date
        self.current_end_date = current_end_date

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.CardGroup([
                    dbc.Card([
                        dbc.CardHeader('Tweet frequency'),
                        dbc.CardBody([
                            self.graph_tweet
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader('User frequency'),
                        dbc.CardBody([
                            self.graph_users
                        ])
                    ]),
                ])
            ], width=12),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, hashtags, start_date, end_date):
        return self.plot_factory.plot_tweet_series(dataset, hashtags, start_date, end_date), \
            self.plot_factory.plot_user_series(dataset, hashtags, start_date, end_date)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_tweet, 'figure'),
            Output(self.graph_users, 'figure'),
            [Input(self.current_dataset, 'data'),
             Input(self.current_hashtags, 'data'),
             Input(self.current_start_date, 'data'),
             Input(self.current_end_date, 'data')],
        )(self.update)


class TopTableComponent(DashComponent):
    def __init__(self, plot_factory, current_dataset, current_hashtags, current_start_date, current_end_date,
                 name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.data = None
        self.current_dataset = current_dataset
        self.current_hashtags = current_hashtags
        self.current_start_date = current_start_date
        self.current_end_date = current_end_date
        self.table = DataTable(data=[], id=f'table-{self.name}',
                               columns=[{"name": i, "id": i} for i in self.plot_factory.top_table_columns],
                               editable=False,
                               filter_action="native",
                               sort_action="native",
                               sort_mode="multi",
                               column_selectable="single",
                               row_selectable="multi",
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

    def extract_hashtag_from_top_table(self, active_cell):
        text = self.data['Text'].iloc[active_cell['row']]
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None

    def callbacks(self, app):
        app.callback(
            Output(self.table, 'data'),
            [Input(self.current_dataset, 'data'),
             Input(self.current_start_date, 'data'),
             Input(self.current_end_date, 'data')],
        )(self.update)
        app.callback(
            Output(self.current_hashtags, 'data', allow_duplicate=True),
            [Input(self.table, 'active_cell')],
        )(self.update_hashtags)


class EgonetComponent(DashComponent):
    def __init__(self, plot_factory, current_dataset, current_user, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.current_dataset = current_dataset
        self.current_user = current_user
        self.available_datasets = plot_factory.available_datasets
        self.graph_egonet = dcc.Graph(figure={}, id=f'fig-{self.name}', style={'height': '70vh'})
        available_users = self.plot_factory.get_users(self.available_datasets[0])
        self.user_dropdown = dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users],
                                          # value=available_users[0],
                                          id=f'user-dropdown-{self.name}')
        self.depth_slider = dcc.Slider(min=1, max=5, step=1, value=2, id=f'slider-{self.name}')

    def layout(self, params=None):
        return dbc.Row([dbc.Col([
            dbc.Row([
                dbc.Col([
                    self.graph_egonet
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.user_dropdown,
                    self.depth_slider
                ]),
            ])],
            width=10)], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, user, depth):
        return self.plot_factory.plot_egonet(dataset, user, depth)

    def callbacks(self, app):
        app.callback(
            Output(self.graph_egonet, 'figure'),
            [Input(self.current_dataset, 'data'),
             Input(self.current_user, 'data'),
             Input(self.depth_slider, 'value')],
        )(self.update)


class ControlPanelComponent(DashComponent):
    def __init__(self, plot_factory, current_dataset, current_hashtags, current_start_date, current_end_date, name=None,
                 max_wordcloud_words=100, wordcloud_width=800, wordcloud_height=400):
        super().__init__(name)
        self.wordcloud_height = wordcloud_height
        self.wordcloud_width = wordcloud_width
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.max_wordcloud_words = max_wordcloud_words
        self.current_dataset = current_dataset
        self.current_hashtags = current_hashtags
        self.current_start_date = current_start_date
        self.current_end_date = current_end_date
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
        return dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Dataset'),
                            dbc.CardBody([
                                self.dataset_dropdown
                            ])
                        ], style={'margin-bottom': '1rem'}),
                    ]),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader('Date range'),
                            dbc.CardBody([
                                self.date_picker
                            ])
                        ]),
                    ]),
                ]),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Hashtag frequency'),
                    dbc.CardBody([
                        self.wordcloud
                    ], style={'width': f'{self.wordcloud_width + 20}px !important'})
                ]),
            ], width=8),

        ], justify='center')

    def callbacks(self, app):

        app.callback(
            Output(self.date_picker, 'min_date_allowed'),
            Output(self.date_picker, 'max_date_allowed'),
            Output(self.date_picker, 'start_date'),
            Output(self.date_picker, 'end_date'),
            [Input(self.current_dataset, 'data')],
        )(self.update_date_range)

        app.callback(
            Output(self.current_dataset, 'data'),
            [Input(self.dataset_dropdown, 'value')],
        )(self.update_dataset_storage)

        app.callback(
            Output(self.current_hashtags, 'data'),
            Input(self.wordcloud, 'click'),
        )(self.update_hashtag_storage)

        app.callback(
            Output(self.current_start_date, 'data'),
            [Input(self.date_picker, 'start_date')],
        )(self.update_start_date_storage)

        app.callback(
            Output(self.current_end_date, 'data'),
            [Input(self.date_picker, 'end_date')],
        )(self.update_end_date_storage)


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, top_table_factory, egonet_plot_factory, name=None,
                 max_wordcloud_words=100):
        super().__init__(name=name)
        self.max_wordcloud_words = max_wordcloud_words

        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.top_table_factory = top_table_factory
        self.available_datasets = tweet_user_plot_factory.available_datasets

        self.current_dataset = dcc.Store(id=f'current-dataset-{self.name}')
        self.current_hashtags = dcc.Store(id=f'current-hashtags-{self.name}')
        self.current_start_date = dcc.Store(id=f'current-start-date-{self.name}')
        self.current_end_date = dcc.Store(id=f'current-end-date-{self.name}')
        self.current_user = dcc.Store(id=f'current-user-{self.name}')

        self.top_table_component = TopTableComponent(top_table_factory,
                                                     current_dataset=self.current_dataset,
                                                     current_hashtags=self.current_hashtags,
                                                     current_start_date=self.current_start_date,
                                                     current_end_date=self.current_end_date,
                                                     name='top')
        self.tweet_user_ts_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory,
                                                                    current_dataset=self.current_dataset,
                                                                    current_hashtags=self.current_hashtags,
                                                                    current_start_date=self.current_start_date,
                                                                    current_end_date=self.current_end_date,
                                                                    name='ts')

        self.egonet_component = EgonetComponent(egonet_plot_factory,
                                                current_dataset=self.current_dataset,
                                                current_user=self.current_user,
                                                name='egonet')

        self.control_panel_component = ControlPanelComponent(tweet_user_plot_factory,
                                                             current_dataset=self.current_dataset,
                                                             current_hashtags=self.current_hashtags,
                                                             current_start_date=self.current_start_date,
                                                             current_end_date=self.current_end_date,
                                                             name='control',
                                                             max_wordcloud_words=self.max_wordcloud_words)

    def layout(self, params=None):
        return dbc.Container([
            self.current_dataset,
            self.current_hashtags,
            self.current_start_date,
            self.current_end_date,
            self.current_user,
            dbc.NavbarSimple(
                brand="REMISS – Towards a methodology to reduce misinformation spread about vulnerable and stigmatised groups",
                brand_href="#",
                sticky="top",
                style={'font-size': '1.5rem', 'font-weight': 'bold', 'margin-bottom': '1rem'},
                fluid=True,

            ),
            html.Div([], style={'margin-bottom': '1rem'}, id=f'placeholder-{self.name}'),
            self.control_panel_component.layout(),
            self.egonet_component.layout(),
            self.top_table_component.layout(),
            self.tweet_user_ts_component.layout(),
        ], fluid=True)

    def update_placeholder(self, dataset, hashtags, start_date, end_date):
        return html.H1(f'Hashtag: {hashtags}, Dataset: {dataset}, Start date: {start_date}, End date: {end_date}')

    def callbacks(self, app):
        app.callback(
            Output(f'placeholder-{self.name}', 'children'),
            [Input(self.current_dataset, 'data'),
             Input(self.current_hashtags, 'data'),
             Input(self.current_start_date, 'data'),
             Input(self.current_end_date, 'data')],
        )(self.update_placeholder)
        self.control_panel_component.callbacks(app)
        self.tweet_user_ts_component.callbacks(app)
        self.top_table_component.callbacks(app)
        self.egonet_component.callbacks(app)
