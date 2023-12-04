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
        raise NotImplementedError()


class TweetUserTimeSeriesComponent(DashComponent):
    def __init__(self, plot_factory, dataset_dropdown, date_picker, wordcloud, top_table, name=None):
        super().__init__(name=name)
        self.top_table = top_table
        self.wordcloud = wordcloud
        self.date_picker = date_picker
        self.dataset_dropdown = dataset_dropdown
        self.plot_factory = plot_factory

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure={}, id=f'fig-tweet-{self.name}')
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id=f'fig-users-{self.name}')
                ]),
            ]),

        ])

    def update_plots(self, dataset, start_date, end_date, click_data, active_cell):
        if callback_context.triggered_id == self.wordcloud and click_data and len(click_data) == 1:
            hashtag = click_data[0][0]
        elif callback_context.triggered_id == self.top_table.id and active_cell:
            hashtag = self.extract_hashtag_from_top_table(active_cell)
        else:
            hashtag = None

        fig_tweet = self.plot_factory.plot_tweet_series(dataset, hashtag, start_date, end_date)
        fig_users = self.plot_factory.plot_user_series(dataset, hashtag, start_date, end_date)
        return fig_tweet, fig_users

    def callbacks(self, app):
        app.callback(
            Output(f'fig-tweet-{self.name}', 'figure'),
            Output(f'fig-users-{self.name}', 'figure'),
            Input(self.dataset_dropdown.id, 'value'),
            Input(self.date_picker.id, 'start_date'),
            Input(self.date_picker.id, 'end_date'),
            Input(self.wordcloud.id, 'list'),
            Input(self.top_table.id, 'active_cell'),
        )(self.update_plots)

    def extract_hashtag_from_top_table(self, active_cell):
        text = self.top_table.data['Text'].iloc[active_cell['row']]
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None


class TopTableComponent(DashComponent):
    def __init__(self, plot_factory, dataset_dropdown, date_picker, name=None):
        super().__init__(name=name)
        self.dataset_dropdown = dataset_dropdown
        self.date_picker = date_picker
        self.plot_factory = plot_factory
        self.data = None
        self.table = DataTable(data=[], id=self.id,
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
                               )

    @property
    def id(self):
        return f'table-{self.name}'

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.table
                ]),
            ]),

        ])

    def update_table(self, dataset, start_date, end_date):
        df_top_table = self.plot_factory.get_top_table(dataset, start_date, end_date)
        self.data = df_top_table
        return df_top_table.to_dict('records')

    def callbacks(self, app):
        app.callback(
            Output(self.id, 'data'),
            Input(self.dataset_dropdown, 'value'),
            Input(self.date_picker, 'start_date'),
            Input(self.date_picker, 'end_date'),
        )(self.update_table)


class EgonetComponent(DashComponent):
    def __init__(self, plot_factory, dataset_dropdown, top_table, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]
        self.dataset_dropdown = dataset_dropdown
        self.top_table = top_table

    def layout(self, params=None):
        available_users = self.plot_factory.get_users(self.current_dataset)
        return dbc.Container([
            dbc.Row([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure={}, id=f'fig-{self.name}')
                    ])
                ]),
                dbc.Col([
                    dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users],
                                 # value=available_users[0],
                                 id=f'user-dropdown-{self.name}'),
                    dcc.Slider(min=1, max=5, step=1, value=2, id=f'slider-{self.name}'),
                ]),
            ]),
        ])

    def update_egonet(self, dataset, user, active_cell, depth):
        if callback_context.triggered_id == self.top_table.id and active_cell:
            user = self.top_table.data['User'].iloc[active_cell['row']]
        return self.plot_factory.plot_egonet(dataset, user, depth)

    def callbacks(self, app):
        app.callback(
            Output(f'fig-{self.name}', 'figure'),
            Input(self.dataset_dropdown.id, 'value'),
            Input(f'user-dropdown-{self.name}', 'value'),
            Input(self.top_table.id, 'active_cell'),
            Input(f'slider-{self.name}', 'value'),
        )(self.update_egonet)


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, top_table_factory, egonet_plot_factory, name=None,
                 max_wordcloud_words=100):
        super().__init__(name=name)
        self.max_wordcloud_words = max_wordcloud_words

        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.top_table_factory = top_table_factory
        self.available_datasets = tweet_user_plot_factory.available_datasets
        self.date_picker = self.get_date_picker_component()
        self.wordcloud = self.get_wordcloud_component()
        self.dataset_dropdown = self.get_dataset_dropdown_component()

        self.top_table_component = TopTableComponent(top_table_factory, dataset_dropdown=self.dataset_dropdown,
                                                     date_picker=self.date_picker, name='top')
        self.tweet_user_ts_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory,
                                                                    dataset_dropdown=self.dataset_dropdown,
                                                                    date_picker=self.date_picker,
                                                                    wordcloud=self.wordcloud,
                                                                    top_table=self.top_table_component,
                                                                    name='ts')

        self.egonet_component = EgonetComponent(egonet_plot_factory, dataset_dropdown=self.dataset_dropdown,
                                                top_table=self.top_table_component,
                                                name='egonet')

    @property
    def dataset_dropdown_id(self):
        return f'dataset-dropdown-{self.name}'

    @property
    def date_picker_id(self):
        return f'date-picker-{self.name}'

    @property
    def wordcloud_id(self):
        return f'wordcloud-{self.name}'

    def get_wordcloud_component(self):
        available_hashtags_freqs = self.tweet_user_plot_factory.get_hashtag_freqs(self.available_datasets[0])
        if self.max_wordcloud_words:
            print(f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
        min_freq = min([x[1] for x in available_hashtags_freqs])

        return DashWordcloud(
            list=available_hashtags_freqs,
            width=1200, height=400,
            rotateRatio=0.5,
            shrinkToFit=True,
            weightFactor=10 / min_freq,
            shape='circle',
            hover=True,
            id=self.wordcloud_id, )

    def get_dataset_dropdown_component(self):
        return dcc.Dropdown(options=[{"label": x, "value": x} for x in self.available_datasets],
                            value=self.available_datasets[0],
                            id=self.dataset_dropdown_id)

    def get_date_picker_component(self):
        min_date_allowed, max_date_allowed, start_date, end_date = self.update_date_range(self.available_datasets[0])
        return dcc.DatePickerRange(
            id=self.date_picker_id,
            min_date_allowed=min_date_allowed,
            max_date_allowed=max_date_allowed,
            initial_visible_month=min_date_allowed,
            start_date=min_date_allowed,
            end_date=max_date_allowed,
        )

    def layout(self, params=None):

        return dbc.Container([
            dbc.Row([
                html.Div('Remiss', className="text-primary text-center fs-3")
            ]),
            dbc.Row([
                dbc.Col([
                    self.dataset_dropdown,
                ]),
                dbc.Col([
                    self.date_picker,

                ]),
                dbc.Col([
                    self.wordcloud,
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.egonet_component.layout(),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.top_table_component.layout(),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.tweet_user_ts_component.layout(),
                ]),
            ]),
        ])

    def update_wordcloud(self, dataset):
        available_hashtags_freqs = self.tweet_user_plot_factory.get_hashtag_freqs(dataset)
        if self.max_wordcloud_words:
            print(f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
        min_freq = min([x[1] for x in available_hashtags_freqs])

        return available_hashtags_freqs, 10 / min_freq

    def update_date_range(self, dataset):
        min_date_allowed, max_date_allowed = self.tweet_user_plot_factory.get_date_range(dataset)
        return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed

    def callbacks(self, app):
        app.callback(
            Output(self.wordcloud.id, 'list'),
            Output(self.wordcloud.id, 'weightFactor'),
            Input(self.dataset_dropdown.id, 'value'),
        )(self.update_wordcloud)
        app.callback(
            Output(self.date_picker.id, 'min_date_allowed'),
            Output(self.date_picker.id, 'max_date_allowed'),
            Output(self.date_picker.id, 'start_date'),
            Output(self.date_picker.id, 'end_date'),
            Input(self.dataset_dropdown.id, 'value'),

        )(self.update_date_range)
        self.tweet_user_ts_component.callbacks(app)
        self.top_table_component.callbacks(app)
        self.egonet_component.callbacks(app)
