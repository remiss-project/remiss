import logging

import dash_bootstrap_components as dbc
from dash import Input, Output, ctx
from dash.dash_table import DataTable

from components.components import RemissComponent

logger = logging.getLogger(__name__)

PAGE_SIZE = 10


class TweetTableComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None,
                 top_table_columns=(
                         'ID', 'User', 'Text', 'Retweets', 'Party', 'Multimodal', 'Profiling',
                         'Suspicious content', 'Cascade size', 'Legitimacy', 'Reputation', 'Status')):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.data = None
        self.displayed_data = None
        self.state = state
        self.top_table_columns = top_table_columns
        self.operators = [['ge ', '>='],
                          ['le ', '<='],
                          ['lt ', '<'],
                          ['gt ', '>'],
                          ['ne ', '!='],
                          ['eq ', '='],
                          ['contains '],
                          ['datestartswith ']]

        self.table = DataTable(data=[], id=f'table-{self.name}',
                               columns=[{"name": i, "id": i} for i in self.top_table_columns],
                               editable=False,
                               cell_selectable=True,
                               row_selectable=False,

                               page_current=0,
                               page_size=PAGE_SIZE,
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
                                   {'if': {'column_id': 'ID'},
                                    'width': '40%'},
                               ]
                               )

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                self.table
            ], width=12),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date, hashtags, page_current, sort_by, filter_query):
        logger.debug(f'Updating tweet table with dataset {dataset}, start date {start_date}, end date {end_date}')
        if ctx.triggered_id in {f'{self.state.current_dataset}.data', f'{self.state.current_start_date}.data',
                                f'{self.state.current_end_date}.data', f'{self.state.current_hashtags}.data'} or self.data is None:
            self.data = self.plot_factory.get_top_table_data(dataset, start_date, end_date, hashtags)
            self.data = self.data.round(2)
            self.data['Multimodal'] = self.data['Multimodal'].apply(lambda x: 'Yes' if x else 'No')
            self.data['Profiling'] = self.data['Profiling'].apply(lambda x: 'Yes' if x else 'No')
            self.data['id'] = self.data['ID']

            page_count = len(self.data) // PAGE_SIZE if self.data is not None else 0
            return self.data.iloc[:, PAGE_SIZE].to_dict(orient='records'), page_count
        else:

            self.displayed_data = self.data
            if filter_query:
                filtering_expressions = filter_query.split(' && ')
                for filter_part in filtering_expressions:
                    col_name, operator, filter_value = self.split_filter_part(filter_part)

                    if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                        # these operators match pandas series operator method names
                        self.displayed_data = self.displayed_data.loc[
                            getattr(self.displayed_data[col_name], operator)(filter_value)]
                    elif operator == 'contains':
                        self.displayed_data = self.displayed_data.loc[
                            self.displayed_data[col_name].str.contains(filter_value)]
                    elif operator == 'datestartswith':
                        # this is a simplification of the front-end filtering logic,
                        # only works with complete fields in standard format
                        self.displayed_data = self.displayed_data.loc[
                            self.displayed_data[col_name].str.startswith(filter_value)]

            if len(sort_by):
                self.displayed_data = self.displayed_data.sort_values(
                    [col['column_id'] for col in sort_by],
                    ascending=[
                        col['direction'] == 'asc'
                        for col in sort_by
                    ],
                    inplace=False
                )

            page_count = len(self.displayed_data) // PAGE_SIZE if self.displayed_data is not None else 0
            start_row = page_current * PAGE_SIZE
            end_row = (page_current + 1) * PAGE_SIZE
            return self.displayed_data.iloc[start_row:end_row].to_dict(orient='records'), page_count

    def split_filter_part(self, filter_part):
        for operator_type in self.operators:
            for operator in operator_type:
                if operator in filter_part:
                    name_part, value_part = filter_part.split(operator, 1)
                    name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                    value_part = value_part.strip()
                    v0 = value_part[0]
                    if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                        value = value_part[1: -1].replace('\\' + v0, v0)
                    else:
                        try:
                            value = float(value_part)
                        except ValueError:
                            value = value_part

                    # word operators need spaces after them in the filter string,
                    # but we don't want these later
                    return name, operator_type[0].strip(), value

        return [None] * 3

    def update_hashtags(self, active_cell):
        if active_cell and active_cell['column_id'] == 'Text':
            text = self.data[self.data['ID'] == active_cell['row_id']]['Text'].values[0]
            hashtags = self.extract_hashtag_from_tweet_text(text)
            logger.debug(f'Updating hashtags state with {hashtags}')
            return hashtags

        return None

    def update_tweet(self, active_cell):
        if active_cell and active_cell['column_id'] == 'ID':
            tweet_id = active_cell['row_id']
            logger.debug(f'Updating tweet state with {tweet_id}')
            return tweet_id

        return None

    def update_user(self, active_cell):
        if active_cell and active_cell['column_id'] == 'User':
            user = self.data[self.data['ID'] == active_cell['row_id']]['Author ID'].values[0]
            logger.debug(f'Updating user state with {user}')
            return user

        return None

    def extract_hashtag_from_tweet_text(self, text):
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None

    def callbacks(self, app):
        app.callback(
            Output(self.table, 'data'),
            Output(self.table, 'page_count'),
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
            Output(self.state.current_hashtags, 'data', allow_duplicate=True),
            [Input(self.table, 'active_cell')],
        )(self.update_hashtags)
        app.callback(
            Output(self.state.current_user, 'data'),
            [Input(self.table, 'active_cell')],
        )(self.update_user)
        app.callback(
            Output(self.state.current_tweet, 'data'),
            [Input(self.table, 'active_cell')],
        )(self.update_tweet)
