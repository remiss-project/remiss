from abc import ABC

import dash_bootstrap_components as dbc
from dash import dcc
import shortuuid
from dash import html, Input, Output, State
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

    def layout(self, params=None):
        raise NotImplementedError()

    def callbacks(self, app):
        raise NotImplementedError()


class TweetUserTimeSeriesComponent(DashComponent):
    def __init__(self, plot_factory, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]

    def layout(self, params=None):
        min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(self.current_dataset)
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(self.current_dataset)
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.DatePickerRange(
                        id=f'evolution-date-picker-range-{self.name}',
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
                    DashWordcloud(
                        list=available_hashtags_freqs,
                        width=1200, height=400,
                        rotateRatio=0.5,
                        shrinkToFit=True,
                        shape='circle',
                        hover=True,
                        id=f'wordcloud-{self.name}'),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure={}, id=f'tweet-evolution-{self.name}')
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id=f'users-evolution-{self.name}')
                ]),
            ]),

        ])

    def callbacks(self, app):
        @app.callback(
            Output(component_id=f'evolution-date-picker-range-{self.name}', component_property='min_date_allowed'),
            Output(component_id=f'evolution-date-picker-range-{self.name}', component_property='max_date_allowed'),
            Output(component_id=f'evolution-date-picker-range-{self.name}', component_property='start_date'),
            Output(component_id=f'evolution-date-picker-range-{self.name}', component_property='end_date'),
            Input(component_id='dropdown-dataset', component_property='value')
        )
        def update_(dataset):
            min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(dataset)
            return min_date_allowed, max_date_allowed, min_date_allowed

        @app.callback(
            Output(component_id=f'wordcloud-{self.name}', component_property='list'),
            Input(component_id=f'dropdown-dataset', component_property='value'),
        )
        def update_wordcloud(dataset):
            available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(dataset)
            return available_hashtags_freqs

        @app.callback(
            Output(component_id=f'tweet-evolution-{self.name}', component_property='figure'),
            Output(component_id=f'users-evolution-{self.name}', component_property='figure'),
            Input(component_id=f'dropdown-dataset', component_property='value'),
            Input(component_id=f'evolution-date-picker-range-{self.name}', component_property='start_date'),
            Input(component_id=f'evolution-date-picker-range-{self.name}', component_property='end_date'),
            Input(component_id=f'wordcloud-{self.name}', component_property='click')

        )
        def update_plots(dataset, start_date, end_date, click_data):
            if click_data:
                hashtag = click_data['points'][0]['text']
            else:
                hashtag = None
            return self.plot_factory.plot_tweet_series(dataset, hashtag, start_date, end_date), \
                self.plot_factory.plot_user_series(dataset, hashtag, start_date, end_date)


class EgonetComponent(DashComponent):
    def __init__(self, plot_factory, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]

    def layout(self, params=None):
        available_users = self.plot_factory.get_users(self.current_dataset)
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users],
                                 value=available_users[0],
                                 id=f'user-dropdown-{self.name}'),
                    dcc.Slider(min=1, max=5, step=1, value=2, id=f'range-slider-{self.name}'),
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id=f'egonet-{self.name}')
                ])
            ]),
        ])

    def callbacks(self, app):
        @app.callback(
            Output(component_id=f'egonet-{self.name}', component_property='figure'),
            Input(component_id='dropdown-dataset', component_property='value'),
            Input(component_id=f'user-dropdown-{self.name}', component_property='value'),
            Input(component_id=f'range-slider-{self.name}', component_property='value'),
        )
        def update_plots(dataset, user, depth):
            return self.plot_factory.plot_egonet(dataset, user, depth)


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, egonet_plot_factory, name=None):
        super().__init__(name=name)
        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.tweet_user_time_series_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory,
                                                                             name='tweet-user-time-series')
        self.egonet_component = EgonetComponent(egonet_plot_factory, name='egonet')

    def layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                html.Div('Remiss', className="text-primary text-center fs-3")
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        options=[{"label": x, "value": x} for x in self.tweet_user_plot_factory.available_datasets],
                        value=self.tweet_user_plot_factory.available_datasets[0],
                        id='dropdown-dataset'),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.tweet_user_time_series_component.layout(),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.egonet_component.layout(),
                ]),
            ]),
        ])

    def callbacks(self, app):
        self.tweet_user_time_series_component.callbacks(app)
        self.egonet_component.callbacks(app)
