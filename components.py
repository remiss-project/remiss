from abc import ABC

import dash_bootstrap_components as dbc
import shortuuid
from dash import dcc, State
from dash import html, Input, Output
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
    def __init__(self, plot_factory, name=None):
        super().__init__(name=name)
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

    def update_plots(self, dataset, start_date, end_date, hashtag):
        fig_tweet = self.plot_factory.plot_tweet_series(dataset, hashtag, start_date, end_date)
        fig_users = self.plot_factory.plot_user_series(dataset, hashtag, start_date, end_date)
        return fig_tweet, fig_users


class TopTableComponent(DashComponent):
    def __init__(self, plot_factory, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    DataTable(data=[], id=f'top-table-{self.name}',
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
                ]),
            ]),

        ])

    def update_table(self, dataset, start_date, end_date):
        df_top_table = self.plot_factory.get_top_table(dataset, start_date, end_date)

        return df_top_table.to_dict('records')


class EgonetComponent(DashComponent):
    def __init__(self, plot_factory, name=None, dataset_dropdown_id=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]
        self.dataset_dropdown = dataset_dropdown_id if dataset_dropdown_id else f'dataset-dropdown-{self.name}'

    def layout(self, params=None):
        available_users = self.plot_factory.get_users(self.current_dataset)
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users],
                                 # value=available_users[0],
                                 id=f'user-dropdown-{self.name}'),
                    dcc.Slider(min=1, max=5, step=1, value=2, id=f'slider-{self.name}'),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure={}, id=f'fig-{self.name}')
                ])
            ]),
        ])

    def update_egonet(self, dataset, user, depth):
        return self.plot_factory.plot_egonet(dataset, user, depth)


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, top_table_factory, egonet_plot_factory, name=None,
                 max_wordcloud_words=100):
        super().__init__(name=name)
        self.max_wordcloud_words = max_wordcloud_words

        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.top_table_factory = top_table_factory
        self.available_datasets = tweet_user_plot_factory.available_datasets

        self.tweet_user_ts_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory, name='ts')

        self.egonet_component = EgonetComponent(egonet_plot_factory, name='egonet')
        self.top_table_component = TopTableComponent(top_table_factory, name='top')

        self.date_picker_component = self.get_date_picker_component(*self.update_date_range(self.available_datasets[0]))
        self.wordcloud_component = self.get_wordcloud_component()
        self.dataset_dropdown_component = self.get_dataset_dropdown_component()

    def get_wordcloud_component(self):
        available_hashtags_freqs = self.tweet_user_plot_factory.get_hashtag_freqs(self.current_dataset)
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
            id=f'wordcloud-{self.name}')

    def get_dataset_dropdown_component(self):
        return dcc.Dropdown(options=[{"label": x, "value": x} for x in self.available_datasets],
                            value=self.available_datasets[0],
                            id=f'dataset-dropdown-{self.name}')

    def get_date_picker_component(self, min_date_allowed, max_date_allowed):
        return dcc.DatePickerRange(
            id=f'date-picker-{self.name}',
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
                    self.dataset_dropdown_component,
                ]),
                dbc.Col([
                    self.date_picker_component,

                ]),
                dbc.Col([
                    self.wordcloud_component,
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
            Input(f'dataset-dropdown-{self.name}', 'value'),
            Output(f'wordcloud-{self.name}', 'list'),
            Output(f'wordcloud-{self.name}', 'weightFactor'),
        )(self.update_wordcloud)
        app.callback(
            Input(f'dataset-dropdown-{self.name}', 'value'),
            Output(f'date-picker-{self.name}', 'min_date_allowed'),
            Output(f'date-picker-{self.name}', 'max_date_allowed'),
            Output(f'date-picker-{self.name}', 'start_date'),
            Output(f'date-picker-{self.name}', 'end_date'),
        )(self.update_date_range)
        self.tweet_user_ts_component.callbacks(app)
