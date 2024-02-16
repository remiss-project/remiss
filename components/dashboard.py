import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from dash_holoniq_wordcloud import DashWordcloud

from components.components import RemissComponent
from components.control_panel import ControlPanelComponent
from components.egonet import EgonetComponent
from components.time_series import TimeSeriesComponent
from components.top_table import TopTableComponent




class RemissState(RemissComponent):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.current_dataset = dcc.Store(id=f'current-dataset-{self.name}', storage_type='session')
        self.current_hashtags = dcc.Store(id=f'current-hashtags-{self.name}', storage_type='session')
        self.current_start_date = dcc.Store(id=f'current-start-date-{self.name}', storage_type='session')
        self.current_end_date = dcc.Store(id=f'current-end-date-{self.name}', storage_type='session')
        self.current_user = dcc.Store(id=f'current-user-{self.name}', storage_type='session')

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


class RemissDashboard(RemissComponent):
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
        self.tweet_user_ts_component = TimeSeriesComponent(tweet_user_plot_factory,
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