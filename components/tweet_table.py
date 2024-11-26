import logging

import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, ctx
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate

from components.components import RemissComponent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TweetTableComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None,
                 top_table_columns=(
                         'ID', 'User', 'Text', 'Retweets', 'Party', 'Multimodal', 'Profiling',
                         'Suspicious content', 'Legitimacy', 'Reputation', 'Status'),
                 page_size=10,
                 anonymous=False):
        super().__init__(name=name)
        self.anonymous = anonymous
        self.page_size = page_size
        self.plot_factory = plot_factory
        self.data = None
        self.displayed_data = None
        self.state = state
        if self.anonymous:
            top_table_columns = [x for x in top_table_columns if x not in ['User', 'Author ID']]
        self.top_table_columns = top_table_columns

        self.table = DataTable(data=[], id=f'table-{self.name}',
                               columns=[{"name": i, "id": i} for i in self.top_table_columns],
                               editable=False,
                               cell_selectable=True,
                               row_selectable=False,

                               page_current=0,
                               page_size=self.page_size,
                               page_action='custom',

                               filter_action='custom',
                               filter_query='',

                               sort_action='custom',
                               sort_mode='multi',
                               sort_by=[],

                               style_cell={
                                   'overflow': 'hidden',
                                   'textOverflow': 'ellipsis',
                                   'maxWidth': 0,
                               },
                               style_cell_conditional=[
                                   {'if': {'column_id': 'Text'},
                                    'width': '30%',
                                    'textOverflow': 'visible',
                                    'whiteSpace': 'normal',
                                    'overflow': 'visible', },
                                   {'if': {'column_id': 'ID'},
                                    'overflow': 'visible',
                                    'textOverflow': 'visible',
                                    'whiteSpace': 'normal',
                                    'width': '10%'},
                               ]
                               )

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Tweets', id=f'title-{self.name}'),
                    dbc.CardBody([
                        dcc.Loading([
                            self.table
                        ])
                    ], style={'height': '100%', 'margin': '0'}),
                    dbc.CardFooter(
                        dbc.Row([
                            dbc.Col([
                                html.H6("Legitimacy"),
                                html.P(
                                    "This metric evaluates users' perceived trustworthiness and acceptance "
                                    "within the network. It reflects the community’s confidence in a user, "
                                    "derived from their interaction history. Legitimacy is measured by the "
                                    "number of times an author's tweets have been interacted with, such as "
                                    "retweets or quotes."
                                )
                            ]),
                            dbc.Col([
                                html.H6("Reputation"),
                                html.P(
                                    "Reputation reflects the collective impression of a user based on their "
                                    "interactions, content quality, and alignment with network trends. It is "
                                    "a composite score indicating the user’s standing within the network. "
                                    "Reputation is calculated as the cumulative summation of Legitimacy over "
                                    "a given time period."
                                )
                            ]),
                            dbc.Col([
                                html.H6("Status"),
                                html.P(
                                    "Status measures a user’s influence and the breadth of their impact across "
                                    "the network. It assesses their ability to reach and engage a broad audience. "
                                    "Status is computed by ranking authors based on the cumulative summation "
                                    "of Legitimacy over time, relative to other users."
                                )
                            ])
                        ])
                    )
                ], style={'height': '100%'}),

            ], width=12),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date, hashtags, page_current, sort_by, filter_query):
        if ctx.triggered_id == self.state.current_dataset.id:
            # The dataset has changed, reset the table controls
            page_current = 0
            sort_by = []
            filter_query = ''

        logger.debug(f'Updating tweet table with '
                     f'dataset {dataset}, start date {start_date}, end date {end_date}, hashtags {hashtags},'
                     f'page {page_current}, sort by {sort_by}, filter query {filter_query}')
        self.data = self.plot_factory.get_tweet_table(dataset, start_date, end_date, hashtags,
                                                      start_tweet=page_current * self.page_size,
                                                      amount=self.page_size,
                                                      sort_by=sort_by,
                                                      filter_query=filter_query)
        self.data = self.data.round(2)
        self.data['Multimodal'] = self.data['Multimodal'].apply(lambda x: 'Yes' if x else 'No')
        self.data['Profiling'] = self.data['Profiling'].apply(lambda x: 'Yes' if x else 'No')
        self.data['id'] = self.data['ID']
        data_to_show = self.data
        if self.anonymous:
            data_to_show = data_to_show.drop(columns=['Author ID', 'User'])
        return data_to_show.to_dict(orient='records')

    def reset_table(self, dataset):
        return 0, '', []

    def update_page_count(self, dataset, start_date, end_date, hashtags):
        size = self.plot_factory.get_tweet_table_size(dataset, start_date, end_date, hashtags)
        return size // self.page_size if size is not None else 0

    def update_hashtags_state(self, active_cell):
        if active_cell and active_cell['column_id'] == 'Text':
            text = self.data[self.data['ID'] == active_cell['row_id']]['Text'].values[0]
            hashtags = self.extract_hashtag_from_tweet_text(text)
            logger.debug(f'Updating hashtags state with {hashtags}')
            return hashtags

        raise PreventUpdate()

    def update_tweet_state(self, active_cell):
        if active_cell and active_cell['column_id'] == 'ID':
            tweet_id = active_cell['row_id']
            logger.debug(f'Updating tweet state with {tweet_id}')
            return tweet_id

        raise PreventUpdate()

    def update_user_state(self, active_cell):
        if active_cell and active_cell['column_id'] == 'User':
            user = self.data[self.data['ID'] == active_cell['row_id']]['Author ID'].values[0]
            logger.debug(f'Updating user state with {user}')
            return user

        raise PreventUpdate()

    def extract_hashtag_from_tweet_text(self, text):
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None

    def callbacks(self, app):
        app.callback(
            Output(self.table, 'data'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(self.state.current_hashtags, 'data'),
             Input(self.table, 'page_current'),
             Input(self.table, 'sort_by'),
             Input(self.table, 'filter_query'),
             ],
        )(self.update)
        app.callback(
            Output(self.table, 'page_count'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(self.state.current_hashtags, 'data'),
             ],
        )(self.update_page_count)

        app.callback(
            Output(self.state.current_hashtags, 'data', allow_duplicate=True),
            [Input(self.table, 'active_cell')],
        )(self.update_hashtags_state)
        app.callback(
            Output(self.state.current_user, 'data'),
            [Input(self.table, 'active_cell')],
        )(self.update_user_state)
        app.callback(
            Output(self.state.current_tweet, 'data'),
            [Input(self.table, 'active_cell')],
        )(self.update_tweet_state)
        app.callback(
            Output(self.table, 'page_current'),
            Output(self.table, 'filter_query'),
            Output(self.table, 'sort_by'),
            [Input(self.state.current_dataset, 'data')],
        )(self.reset_table)

