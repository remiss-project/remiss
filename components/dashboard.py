import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from components.components import RemissComponent
from components.control_panel import ControlPanelComponent
from components.egonet import EgonetComponent
from components.multimodal import MultimodalComponent
from components.profiling import RadarplotEmotionsComponent, VerticalBarplotPolarityComponent, \
    DonutPlotBehaviour1Component, \
    DonutPlotBehaviour2Component, ProfilingComponent
from components.propagation import PropagationComponent, CascadeCcdfComponent, CascadeCountOverTimeComponent
from components.textual import EmotionPerHourComponent, AverageEmotionBarComponent
from components.time_series import TimeSeriesComponent
from components.tweet_table import TweetTableComponent


class RemissState(RemissComponent):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.current_dataset = dcc.Store(id=f'current-dataset-{self.name}', storage_type='session')
        self.current_hashtags = dcc.Store(id=f'current-hashtags-{self.name}', storage_type='session')
        self.current_start_date = dcc.Store(id=f'current-start-date-{self.name}', storage_type='session')
        self.current_end_date = dcc.Store(id=f'current-end-date-{self.name}', storage_type='session')
        self.current_user = dcc.Store(id=f'current-user-{self.name}', storage_type='session')
        self.current_tweet = dcc.Store(id=f'current-tweet-{self.name}', storage_type='session')

    def layout(self, params=None):
        return html.Div([
            self.current_dataset,
            self.current_hashtags,
            self.current_start_date,
            self.current_end_date,
            self.current_user,
            self.current_tweet,
        ])

    def callbacks(self, app):
        pass


class GeneralPlotsComponent(RemissComponent):
    """
    Includes all the plots that display static information about the dataset.
    - Propagation ccdf and cascade count over time
    - Average emotion barplot
    - Emotion per hour
    """

    def __init__(self, propagation_factory, profile_factory, textual_factory, state, name=None):
        super().__init__(name=name)
        self.cascade_cddf = CascadeCcdfComponent(propagation_factory, state, name=f'cascade-ccdf-{self.name}')
        self.cascade_count_over_time = CascadeCountOverTimeComponent(propagation_factory, state,
                                                                     name=f'cascade-count-over-time-{self.name}')
        self.average_emotion_barplot = AverageEmotionBarComponent(textual_factory, state,
                                                                  name=f'average-emotion-{self.name}')
        self.emotion_per_hour = EmotionPerHourComponent(textual_factory, state, name=f'emotion-per-hour-{self.name}')

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                self.cascade_cddf.layout(params)
            ]),
            dbc.Col([
                self.cascade_count_over_time.layout(params)
            ]),
            dbc.Col([
                self.average_emotion_barplot.layout(params)
            ]),
            dbc.Col([
                self.emotion_per_hour.layout(params)
            ]),
        ])

    def callbacks(self, app):
        self.cascade_cddf.callbacks(app)
        self.cascade_count_over_time.callbacks(app)
        self.average_emotion_barplot.callbacks(app)
        self.emotion_per_hour.callbacks(app)


class FilterablePlotsComponent(RemissComponent):
    """
    Includes all the plots that can be filtered by user, date, etc.
    - Time series
    - Radarplot emotions
    - Vertical barplot polarity
    - Donut plot behaviour 1
    - Donut plot behaviour 2
    - Multimodal
    - Propagation
    """

    def __init__(self, tweet_user_plot_factory,
                 textual_plot_factory,
                 profile_plot_factory,
                 multimodal_plot_factory,
                 propagation_plot_factory,
                 state, name=None):
        super().__init__(name=name)
        self.state = state
        self.time_series = TimeSeriesComponent(tweet_user_plot_factory, state, name=f'time-series-{self.name}')
        self.profiling_component = ProfilingComponent(profile_plot_factory, state, name=f'profiling-{self.name}')

        self.multimodal = MultimodalComponent(multimodal_plot_factory, state, name=f'multimodal-{self.name}')

        self.propagation = PropagationComponent(propagation_plot_factory, state,
                                                name=f'propagation-{self.name}')

    def layout(self, params=None):
        """
        |                   Time series                    |
        | Radarplot emotions | Vertical barplot polarity   |
        | Donut plot behaviour 1 | Donut plot behaviour 2  |
        |                   Multimodal                     |
        |                   Propagation                    |

        :param params:
        :return:
        """
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    self.time_series.layout(params)
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.profiling_component.layout(params)
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.multimodal.layout(params)
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.propagation.layout(params)
                ]),
            ]),
        ])

    def callbacks(self, app):
        self.time_series.callbacks(app)
        self.profiling_component.callbacks(app)
        self.multimodal.callbacks(app)
        self.propagation.callbacks(app)


class RemissDashboard(RemissComponent):
    """
    Main dashboard component.
    | Dataset selector                  |
    | General plots                     |
    | Date range        |     Egonet    |
    | Hashtag wordcloud |               |
    | Tweet table                       |
    | filterable plots                  |

    """

    def __init__(self, tweet_user_plot_factory,
                 tweet_table_factory,
                 propagation_factory,
                 textual_factory,
                 profile_factory,
                 multimodal_factory,
                 name=None,
                 max_wordcloud_words=50, wordcloud_width=400, wordcloud_height=400, match_wordcloud_width=False,

                 debug=False):
        super().__init__(name=name)
        self.debug = debug
        self.available_datasets = tweet_user_plot_factory.available_datasets
        self.state = RemissState(name='state')

        self.dataset_dropdown = self.get_dataset_dropdown_component()
        self.general_plots_component = GeneralPlotsComponent(propagation_factory, profile_factory, textual_factory,
                                                             self.state, name=f'general-plots-{self.name}')
        self.control_panel_component = ControlPanelComponent(tweet_user_plot_factory, self.state,
                                                             name=f'control-panel-{self.name}',
                                                             max_wordcloud_words=max_wordcloud_words,
                                                             wordcloud_width=wordcloud_width,
                                                             wordcloud_height=wordcloud_height,
                                                             match_wordcloud_width=match_wordcloud_width)
        self.egonet_component = EgonetComponent(propagation_factory, self.state, name=f'egonet-{self.name}')
        self.tweet_table_component = TweetTableComponent(tweet_table_factory, self.state,
                                                         name=f'tweet-table-{self.name}')
        self.filterable_plots_component = FilterablePlotsComponent(tweet_user_plot_factory,
                                                                   textual_factory,
                                                                   profile_factory,
                                                                   multimodal_factory,
                                                                   propagation_factory,
                                                                   self.state, name=f'filterable-plots-{self.name}')

    def get_dataset_dropdown_component(self):
        available_datasets = {db_key: db_key.replace('_', ' ').capitalize().strip() for db_key in
                              self.available_datasets}
        return dcc.Dropdown(options=[{"label": db_key, "value": name} for name, db_key in available_datasets.items()],
                            value=self.available_datasets[0],
                            id=f'dataset-dropdown-{self.name}')

    def update_placeholder(self, dataset, hashtags, start_date, end_date, current_user, current_tweet):
        return html.H1(f'Hashtag: {hashtags}, Dataset: {dataset}, Start date: {start_date}, '
                       f'End date: {end_date}, Current user: {current_user}, Current tweet: {current_tweet}')

    def update_dataset_storage(self, dropdown_dataset):
        return dropdown_dataset

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
                    self.dataset_dropdown
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.general_plots_component.layout(params)
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.control_panel_component.layout(params)
                ]),
                dbc.Col([
                    self.egonet_component.layout(params)
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.tweet_table_component.layout(params)
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    self.filterable_plots_component.layout(params)
                ]),
            ]),

        ], fluid=False)

    def callbacks(self, app):
        if self.debug:
            app.callback(
                Output(f'placeholder-{self.name}', 'children'),
                [Input(self.state.current_dataset, 'data'),
                 Input(self.state.current_hashtags, 'data'),
                 Input(self.state.current_start_date, 'data'),
                 Input(self.state.current_end_date, 'data'),
                 Input(self.state.current_user, 'data'),
                 Input(self.state.current_tweet, 'data')
                 ],
            )(self.update_placeholder)
        # Dataset dropdown
        app.callback(
            Output(self.state.current_dataset, 'data'),
            [Input(f'dataset-dropdown-{self.name}', 'value')],
        )(self.update_dataset_storage)

        self.general_plots_component.callbacks(app)
        self.control_panel_component.callbacks(app)
        self.egonet_component.callbacks(app)
        self.tweet_table_component.callbacks(app)
        self.filterable_plots_component.callbacks(app)
