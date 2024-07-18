import json
import logging

import dash_bootstrap_components as dbc
from dash import dcc, Input, Output
from dash_holoniq_wordcloud import DashWordcloud

from components.components import RemissComponent

logger = logging.getLogger(__name__)


class ControlPanelComponent(RemissComponent):
    def __init__(self, plot_factory, state, name=None,
                 max_wordcloud_words=50, wordcloud_width=800, wordcloud_height=400, match_wordcloud_width=False):
        super().__init__(name)
        self.state = state
        self.wordcloud_height = wordcloud_height
        self.wordcloud_width = wordcloud_width
        self.match_wordcloud_width = match_wordcloud_width
        self.plot_factory = plot_factory
        self.available_datasets = plot_factory.available_datasets
        self.max_wordcloud_words = max_wordcloud_words
        self.date_picker = self.get_date_picker_component()
        self.wordcloud = self.get_wordcloud_component()

    def get_wordcloud_component(self):
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(self.available_datasets[0])
        if available_hashtags_freqs:
            if self.max_wordcloud_words:
                logger.info(
                    f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
                available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
            min_freq = min([x[1] for x in available_hashtags_freqs])
        else:
            min_freq = 1
            available_hashtags_freqs = [('No hashtags found', 1)]

        return DashWordcloud(
            list=available_hashtags_freqs,
            width=self.wordcloud_width, height=self.wordcloud_height,
            rotateRatio=0.5,
            shrinkToFit=True,
            weightFactor=10 / min_freq,
            hover=True,
            id=f'wordcloud-{self.name}'
            ,
        )

    def get_date_picker_component(self):
        min_date_allowed, max_date_allowed, start_date, end_date = self.update_date_range(self.available_datasets[0])
        return dcc.DatePickerRange(
            id=f'date-picker-{self.name}',
            min_date_allowed=min_date_allowed,
            max_date_allowed=max_date_allowed,
            initial_visible_month=min_date_allowed,
            start_date=min_date_allowed,
            end_date=max_date_allowed)

    def update_wordcloud(self, dataset):
        logger.info(f'Updating wordcloud with dataset {dataset}')
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(dataset)
        if self.max_wordcloud_words:
            logger.info(
                f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
        min_freq = min([x[1] for x in available_hashtags_freqs])

        return available_hashtags_freqs, 10 / min_freq

    def update_date_range(self, dataset):
        logger.info(f'Updating date range with dataset {dataset}')
        min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(dataset)
        return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed

    def update_dataset_storage(self, dropdown_dataset):
        logger.info(f'Updating dataset storage with {dropdown_dataset}')
        return dropdown_dataset

    def update_hashtag_storage(self, click_data):
        logger.info(f'Updating hashtag storage with {click_data}')
        return [click_data[0]] if click_data else None

    def update_start_date_storage(self, start_date):
        logger.info(f'Updating start date storage with {start_date}')
        return start_date

    def update_end_date_storage(self, end_date):
        logger.info(f'Updating end date storage with {end_date}')
        return end_date

    def layout(self, params=None):
        return dbc.Stack([
            dbc.Card([
                dbc.CardHeader('Date range'),
                dbc.CardBody([
                    self.date_picker
                ])
            ]),

            dbc.Card([
                dbc.CardHeader('Hashtags'),
                dbc.CardBody([
                    dcc.Loading(id=f'loading-wordcloud-{self.name}',
                                type='default',
                                children=self.wordcloud)
                ])
            ]),
        ], gap=2, style={'width': f'{self.wordcloud_width + 40}px'} if self.match_wordcloud_width else None)

    def callbacks(self, app):
        app.callback(
            Output(self.date_picker, 'min_date_allowed'),
            Output(self.date_picker, 'max_date_allowed'),
            Output(self.date_picker, 'start_date'),
            Output(self.date_picker, 'end_date'),
            [Input(self.state.current_dataset, 'data')],
        )(self.update_date_range)

        app.callback(
            Output(self.state.current_hashtags, 'data'),
            Input(self.wordcloud, 'click'),
        )(self.update_hashtag_storage)

        app.callback(
            Output(self.state.current_start_date, 'data'),
            [Input(self.date_picker, 'start_date')],
        )(self.update_start_date_storage)

        app.callback(
            Output(self.state.current_end_date, 'data'),
            [Input(self.date_picker, 'end_date')],
        )(self.update_end_date_storage)
