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
        layout = self._layout(params)
        patch_layout_ids(layout, self.name)
        return layout

    def _layout(self, params=None):
        raise NotImplementedError()

    def callbacks(self, app):
        raise NotImplementedError()

    def id(self, component_id):
        return component_id + '-' + self.name

    def Input(self, component_id, component_property):
        return Input(f'{component_id}-{self.name}', component_property)

    def Output(self, component_id, component_property):
        return Output(f'{component_id}-{self.name}', component_property)

    def State(self, component_id, component_property):
        return State(f'{component_id}-{self.name}', component_property)


class TweetUserTimeSeriesComponent(DashComponent):
    def __init__(self, plot_factory, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]

    def _layout(self, params=None):
        min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(self.current_dataset)
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(self.current_dataset)
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(options=[{"label": x, "value": x} for x in self.available_datasets],
                                 value=self.available_datasets[0],
                                 id='dataset-dropdown'),
                    dcc.DatePickerRange(
                        id='evolution-date-picker-range',
                        min_date_allowed=min_date_allowed,
                        max_date_allowed=max_date_allowed,
                        initial_visible_month=min_date_allowed,
                        start_date=min_date_allowed,
                        end_date=max_date_allowed,
                        display_format='DD/MM/YYYY',
                    ),
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id='tweet-evolution')
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    DashWordcloud(
                        list=available_hashtags_freqs,
                        width=600, height=400,
                        rotateRatio=0.5,
                        shrinkToFit=True,
                        shape='circle',
                        hover=True,
                        id='wordcloud'),
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id='users-evolution')
                ]),
            ]),

        ])

    def callbacks(self, app):
        @app.callback(
            self.Output(component_id='temporal-evolution-date-picker-range', component_property='min_date_allowed'),
            self.Output(component_id='temporal-evolution-date-picker-range', component_property='max_date_allowed'),
            self.Output(component_id='temporal-evolution-date-picker-range', component_property='start_date'),
            self.Output(component_id='temporal-evolution-date-picker-range', component_property='end_date'),
            self.Output(component_id='wordcloud', component_property='list'),
            self.Input(component_id='dropdown-dataset', component_property='value')
        )
        def update_dataset(dataset):
            min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(dataset)
            available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(dataset)
            return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed, available_hashtags_freqs

        @app.callback(
            self.Output(component_id=f'tweet-evolution', component_property='figure'),
            self.Output(component_id=f'users-evolution', component_property='figure'),
            self.Input(component_id=f'dropdown-dataset', component_property='value'),
            self.Input(component_id=f'evolution-date-picker-range', component_property='start_date'),
            self.Input(component_id=f'evolution-date-picker-range', component_property='end_date'),
            self.Input(component_id=f'wordcloud', component_property='click')

        )
        def update_plots(dataset, start_date, end_date, click_data):
            if click_data:
                hashtag = click_data['points'][0]['text']
            else:
                hashtag = None
            return self.plot_factory.get_temporal_evolution(dataset, hashtag, start_date, end_date), \
                self.plot_factory.get_users_evolution(dataset, hashtag, start_date, end_date)


class EgonetComponent(DashComponent):
    def __init__(self, plot_factory, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]

    def _layout(self, params=None):
        available_users = self.plot_factory.get_users(self.current_dataset)
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(options=[{"label": x, "value": x} for x in self.available_datasets],
                                 value=self.available_datasets[0],
                                 id='dataset-dropdown'),
                    dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users],
                                 value=available_users[0],
                                 id='user-dropdown'),
                    dcc.Slider(min=1, max=5, step=1, value=2, id='range-slider'),
                ]),
                dbc.Col([
                    dcc.Graph(figure={}, id='egonet')
                ])
            ]),
        ])

    def callbacks(self, app):
        @app.callback(
            self.Output(component_id=f'egonet', component_property='figure'),
            self.Input(component_id='dropdown-dataset', component_property='value'),
            self.Input(component_id='dropdown-user', component_property='value'),
            self.Input(component_id='range-slider', component_property='value'),
        )
        def update_plots(dataset, user, depth):
            return self.plot_factory.get_egonet(dataset, user, depth)


class RemissDashboard(DashComponent):
    def __init__(self, tweet_user_plot_factory, egonet_plot_factory, name=None):
        super().__init__(name=name)
        self.tweet_user_plot_factory = tweet_user_plot_factory
        self.egonet_plot_factory = egonet_plot_factory
        self.tweet_user_time_series_component = TweetUserTimeSeriesComponent(tweet_user_plot_factory,
                                                                             name='tweet-user-time-series')
        self.egonet_component = EgonetComponent(egonet_plot_factory, name='egonet')

    def _layout(self, params=None):
        return dbc.Container([
            dbc.Row([
                html.Div('Remiss', className="text-primary text-center fs-3")
            ]),
            dbc.Row([
                dbc.Col([
                    self.tweet_user_time_series_component.layout(),
                ]),
                dbc.Col([
                    self.egonet_component.layout(),
                ]),
            ]),
        ])
