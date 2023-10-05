from dash import html
from dash.exceptions import PreventUpdate
from dash_holoniq_wordcloud import DashWordcloud
from dash_oop_components import DashComponent
import dash_bootstrap_components as dbc
import dash_core_components as dcc


class TweetUserTimeSeries(DashComponent):
    def __init__(self, plot_factory):
        super().__init__()
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.current_dataset = self.available_datasets[0]

    def layout(self, params=None):
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

    def component_callbacks(self, app):
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
            self.Output(component_id='tweet-evolution', component_property='figure'),
            self.Output(component_id='users-evolution', component_property='figure'),
            self.Input(component_id='dropdown-dataset', component_property='value'),
            self.Input(component_id='evolution-date-picker-range', component_property='start_date'),
            self.Input(component_id='evolution-date-picker-range', component_property='end_date'),
            self.Input(component_id='wordcloud', component_property='click')

        )
        def update_plots(dataset, start_date, end_date, click_data):
            if not dataset:
                raise PreventUpdate
            if click_data:
                hashtag = click_data['points'][0]['text']
            else:
                hashtag = None
            return self.plot_factory.get_temporal_evolution(dataset, hashtag, start_date, end_date), \
                   self.plot_factory.get_users_evolution(dataset, hashtag, start_date, end_date)