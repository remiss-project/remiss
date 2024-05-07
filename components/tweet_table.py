import dash_bootstrap_components as dbc
from dash import Input, Output
from dash.dash_table import DataTable

from components.components import RemissComponent


class TweetTableComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.data = None
        self.state = state
        self.table = DataTable(data=[], id=f'table-{self.name}',
                               columns=[{"name": i, "id": i} for i in self.plot_factory.top_table_columns],
                               editable=False,
                               filter_action="native",
                               sort_action="native",
                               sort_mode="multi",
                               # column_selectable="multi",
                               # row_selectable="single",
                               row_deletable=False,
                               selected_columns=[],
                               selected_rows=[],
                               page_action="native",
                               page_current=0,
                               page_size=10,
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
        self.only_profiling_checkbox = dbc.Checklist(
            options=[{"label": "Only profiling", "value": "only_profiling"}],
            value=[],
            id=f'only_profiling-{self.name}',
            inline=True,
        )
        self.only_multimodal_checkbox = dbc.Checklist(
            options=[{"label": "Only multimodal", "value": "only_multimodal"}],
            value=[],
            id=f'only_multimodal-{self.name}',
            inline=True,
        )

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                self.only_profiling_checkbox,
                self.only_multimodal_checkbox,
                self.table
            ], width=12),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, start_date, end_date, only_multimodal, only_profiling):
        self.data = self.plot_factory.get_top_table_data(dataset, start_date, end_date, only_multimodal, only_profiling)
        return self.data.to_dict('records')

    def update_hashtags(self, active_cell):
        if active_cell:
            hashtags = self.extract_hashtag_from_top_table(active_cell)
            if hashtags:
                return hashtags
        return None

    def update_user(self, active_cell):
        if active_cell:
            user = self.data['User'].iloc[active_cell['row']]
            return user
        return None

    def extract_hashtag_from_top_table(self, active_cell):
        text = self.data['Text'].iloc[active_cell['row']]
        hashtags = [x[1:] for x in text.split() if x.startswith('#')]
        return hashtags if hashtags else None

    def callbacks(self, app):
        app.callback(
            Output(self.table, 'data'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_start_date, 'data'),
             Input(self.state.current_end_date, 'data'),
             Input(f'only_profiling-{self.name}', 'value'),
             Input(f'only_multimodal-{self.name}', 'value')],
        )(self.update)
        app.callback(
            Output(self.state.current_hashtags, 'data', allow_duplicate=True),
            [Input(self.table, 'active_cell')],
        )(self.update_hashtags)
        app.callback(
            Output(self.state.current_user, 'data'),
            [Input(self.table, 'active_cell')],
        )(self.update_user)
