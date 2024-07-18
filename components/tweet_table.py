import logging

import dash_bootstrap_components as dbc
from dash import Input, Output
from dash.dash_table import DataTable

from components.components import RemissComponent

logger = logging.getLogger(__name__)


class TweetTableComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None,
                 top_table_columns=(
                         'ID', 'User', 'Text', 'Retweets', 'Party', 'Multimodal', 'Profiling', 'Credibility',
                         'Fakeness')):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.data = None
        self.state = state
        self.top_table_columns = top_table_columns

        self.table = DataTable(data=[], id=f'table-{self.name}',
                               columns=[{"name": i, "id": i} for i in self.top_table_columns],
                               editable=False,
                               filter_action="native",
                               sort_action="native",
                               sort_mode="multi",
                               cell_selectable=False,
                               # # column_selectable="multi",
                               row_selectable="single",
                               # row_deletable=False,
                               # selected_columns=[],
                               # selected_rows=[],
                               page_action="native",
                               page_current=0,
                               page_size=20,
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
        logger.info(f'Updating tweet table with dataset {dataset}, start date {start_date}, end date {end_date}')
        self.data = self.plot_factory.get_top_table_data(dataset, start_date, end_date)
        self.data['Multimodal'] = self.data['Multimodal'].apply(lambda x: '✓' if x else '✗')
        self.data['Profiling'] = self.data['Profiling'].apply(lambda x: '✓' if x else '✗')
        return self.data.to_dict('records')

    def update_hashtags(self, selected_rows):
        if selected_rows:
            hashtags = self.extract_hashtag_from_top_table(selected_rows[0])
            logger.info(f'Updating hashtags state with {hashtags}')
            if hashtags:
                return hashtags
        return None

    def update_tweet(self, selected_rows):
        if selected_rows:
            tweet = self.data['ID'].iloc[selected_rows[0]]
            logger.info(f'Updating tweet state with {tweet}')
            return tweet
        return None

    def update_user(self, selected_rows):
        if selected_rows:
            user = self.data['Author ID'].iloc[selected_rows[0]]
            logger.info(f'Updating user state with {user}')
            return user
        return None

    def extract_hashtag_from_top_table(self, selected_row):
        text = self.data['Text'].iloc[selected_row]
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None

    def highlight_row(self, active_cell):
        if active_cell:
            return [{
                'if': {'row_index': active_cell['row']},
                'backgroundColor': 'rgba(0, 0, 0, 0.1)'
            }]
        return []

    def callbacks(self, app):
        app.callback(
            Output(self.table, 'data'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data')],
        )(self.update)
        app.callback(
            Output(self.state.current_hashtags, 'data', allow_duplicate=True),
            [Input(self.table, 'selected_rows')],
        )(self.update_hashtags)
        app.callback(
            Output(self.state.current_user, 'data'),
            [Input(self.table, 'selected_rows')],
        )(self.update_user)
        app.callback(
            Output(self.state.current_tweet, 'data'),
            [Input(self.table, 'selected_rows')],
        )(self.update_tweet)
        app.callback(
            Output(self.table, 'style_data_conditional'),
            [Input(self.table, 'active_cell')],
        )(self.highlight_row)
