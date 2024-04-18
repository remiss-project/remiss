from pathlib import Path

import dash_bootstrap_components as dbc
from dash import dcc, html
from components.components import RemissComponent


class FactCheckingComponent(RemissComponent):
    def __init__(self, plot_factory, state,
                 name=None):
        super().__init__(name=name)
        self.fact_checking_iframe = html.Iframe(src=Path('./../fact_checking_data/test_dataset/47.htm').resolve(), width='100%',
                                                height='800', id=f'fact-checking-iframe-{self.name}')
        self.plot_factory = plot_factory
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('Tweet fact check'),
                    dbc.CardBody([
                        self.fact_checking_iframe
                    ])
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, dataset, hashtags, start_date, end_date):
        pass

    def callbacks(self, app):
        pass
