from abc import ABC

import dash_bootstrap_components as dbc
import shortuuid
from dash import dcc
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
    def __init__(self, plot_factory, name=None, dataset_dropdown_id=None, date_picker_id=None, max_wordcloud_words=100):
        super().__init__(name=name)
        self.max_wordcloud_words = max_wordcloud_words
        self.dataset_dropdown_id = dataset_dropdown_id if dataset_dropdown_id else f'dataset-dropdown-{self.name}'
        self.date_picker_id = date_picker_id if date_picker_id else f'date-picker-{self.name}'
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]

    def layout(self, params=None):
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(self.current_dataset)
        min_freq = min([x[1] for x in available_hashtags_freqs])
        if self.max_wordcloud_words:
            print(f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]

        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    DashWordcloud(
                        list=available_hashtags_freqs,
                        width=1200, height=400,
                        rotateRatio=0.5,
                        shrinkToFit=True,
                        weightFactor=10 / min_freq,
                        shape='circle',
                        hover=True,
                        id=f'wordcloud-{self.name}'),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure={}, id=f'fig-tweet-{self.name}')
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id=f'fig-users-{self.name}')
                ]),
            ]),

        ])

    def update_wordcloud(self, dataset):
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(dataset)
        return available_hashtags_freqs

    def update_date_picker(self, dataset):
        min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(dataset)
        return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed

    def update_plots(self, dataset, start_date, end_date, click_data):
        if click_data:
            hashtag = click_data[0]
        else:
            hashtag = None
        fig_tweet = self.plot_factory.plot_tweet_series(dataset, hashtag, start_date, end_date)
        fig_users = self.plot_factory.plot_user_series(dataset, hashtag, start_date, end_date)
        return fig_tweet, fig_users

    def callbacks(self, app):
        app.callback(
            Output(component_id=self.date_picker_id, component_property='min_date_allowed'),
            Output(component_id=self.date_picker_id, component_property='max_date_allowed'),
            Output(component_id=self.date_picker_id, component_property='start_date'),
            Output(component_id=self.date_picker_id, component_property='end_date'),
            Input(component_id=self.dataset_dropdown_id, component_property='value')
        )(self.update_date_picker)

        app.callback(
            Output(component_id=f'wordcloud-{self.name}', component_property='list'),
            Input(component_id=self.dataset_dropdown_id, component_property='value'),
        )(self.update_wordcloud)

        app.callback(
            Output(component_id=f'fig-tweet-{self.name}', component_property='figure'),
            Output(component_id=f'fig-users-{self.name}', component_property='figure'),
            Input(component_id=self.dataset_dropdown_id, component_property='value'),
            Input(component_id=self.date_picker_id, component_property='start_date'),
            Input(component_id=self.date_picker_id, component_property='end_date'),
            Input(component_id=f'wordcloud-{self.name}', component_property='click')

        )(self.update_plots)


class TopTableComponent(DashComponent):
    def __init__(self, plot_factory, name=None, dataset_dropdown_id=None, date_picker_id=None):
        super().__init__(name=name)
        self.dataset_dropdown_id = dataset_dropdown_id if dataset_dropdown_id else f'dataset_dropdown-{self.name}'
        self.date_picker_id = date_picker_id if date_picker_id else f'date-picker-{self.name}'
        self.plot_factory = plot_factory

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    DataTable(data=[], id=f'table-top-tweets-{self.name}',
                              columns=[{"name": i, "id": i} for i in self.plot_factory.retweeted_table_columns])
                ]),
                dbc.Col([
                    DataTable(data=[], id=f'table-top-users-{self.name}',
                              columns=[{"name": i, "id": i} for i in self.plot_factory.user_table_columns])
                ]),
            ]),

        ])

    def update_table(self, dataset, start_date, end_date):
        df_top_tweets = self.plot_factory.get_top_retweeted(dataset, start_date, end_date)
        df_top_users = self.plot_factory.get_top_users(dataset, start_date, end_date)

        return df_top_tweets.to_dict('records'), df_top_users.to_dict('records')

    def callbacks(self, app):
        app.callback(
            Output(component_id=f'table-top-tweets-{self.name}', component_property='data'),
            Output(component_id=f'table-top-users-{self.name}', component_property='data'),
            Input(component_id=self.dataset_dropdown_id, component_property='value'),
            Input(component_id=self.date_picker_id, component_property='start_date'),
            Input(component_id=self.date_picker_id, component_property='end_date'),
        )(self.update_table)


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

    def callbacks(self, app):
        app.callback(
            Output(component_id=f'fig-{self.name}', component_property='figure'),
            Input(component_id=self.dataset_dropdown, component_property='value'),
            Input(component_id=f'user-dropdown-{self.name}', component_property='value'),
            Input(component_id=f'slider-{self.name}', component_property='value'),
        )(self.update_egonet)


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, top_table_factory, egonet_plot_factory, name=None,
                 dataset_dropdown_id=None, date_picker_id=None):
        super().__init__(name=name)
        self.date_picker_id = date_picker_id if date_picker_id else f'date-picker-{self.name}'
        self.dataset_dropdown_id = dataset_dropdown_id if dataset_dropdown_id else f'dataset-dropdown-{self.name}'
        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.top_table_factory = top_table_factory
        self.tweet_user_ts_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory,
                                                                    dataset_dropdown_id=self.dataset_dropdown_id,
                                                                    date_picker_id=self.date_picker_id,
                                                                    name='ts')

        self.egonet_component = EgonetComponent(egonet_plot_factory, name='egonet',
                                                dataset_dropdown_id=self.dataset_dropdown_id)
        self.top_table_component = TopTableComponent(top_table_factory,
                                                     dataset_dropdown_id=self.dataset_dropdown_id,
                                                     date_picker_id=self.date_picker_id,
                                                     name='top')

    def layout(self, params=None):
        default_dataset = self.tweet_user_plot_factory.available_datasets[0]
        min_date_allowed, max_date_allowed = self.tweet_user_plot_factory.get_date_range(default_dataset)

        return dbc.Container([
            dbc.Row([
                html.Div('Remiss', className="text-primary text-center fs-3")
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        options=[{"label": x, "value": x} for x in self.tweet_user_plot_factory.available_datasets],
                        value=default_dataset,
                        id=f'dataset-dropdown-{self.name}'),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.DatePickerRange(
                        id=f'date-picker-{self.name}',
                        min_date_allowed=min_date_allowed,
                        max_date_allowed=max_date_allowed,
                        initial_visible_month=min_date_allowed,
                        start_date=min_date_allowed,
                        end_date=max_date_allowed,
                        display_format='DD/MM/YYYY',
                    ),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.tweet_user_ts_component.layout(),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.top_table_component.layout(),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.egonet_component.layout(),
                ]),
            ]),
        ])

    def callbacks(self, app):
        self.tweet_user_ts_component.callbacks(app)
        self.top_table_component.callbacks(app)
        self.egonet_component.callbacks(app)
