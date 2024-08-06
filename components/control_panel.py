import json
import logging

import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, html
from dash.exceptions import PreventUpdate
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
        self.filter_display = FilterDisplayComponent(state, name=f'filter-display-{self.name}')

    @property
    def reset_button(self):
        return self.filter_display.reset_button

    def get_wordcloud_component(self):
        available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(self.available_datasets[0])
        if available_hashtags_freqs:
            if self.max_wordcloud_words:
                logger.debug(
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

    def update_wordcloud(self, dataset, author_id):
        if author_id:
            logger.debug(f'Updating wordcloud with dataset {dataset} and user {author_id}')
            available_hashtags_freqs = self.plot_factory.get_hashtags_for_user(dataset, author_id)
        else:
            logger.debug(f'Updating wordcloud with dataset {dataset}')
            available_hashtags_freqs = self.plot_factory.get_hashtag_freqs(dataset)
        if self.max_wordcloud_words < len(available_hashtags_freqs):
            logger.debug(
                f'Using {self.max_wordcloud_words} most frequent hashtags out of {len(available_hashtags_freqs)}.')
            available_hashtags_freqs = available_hashtags_freqs[:self.max_wordcloud_words]
        if available_hashtags_freqs:
            min_freq = min([x[1] for x in available_hashtags_freqs])

            return available_hashtags_freqs, 10 / min_freq
        else:
            return [('No hashtags found', 1)], 1

    def update_date_range(self, dataset):
        logger.debug(f'Updating date range with dataset {dataset}')
        min_date_allowed, max_date_allowed = self.plot_factory.get_date_range(dataset)
        return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed

    def update_current_values_display(self, tweet, hashtags, user):
        logger.debug(f'Updating current values display with tweet {tweet}, hashtags {hashtags}, user {user}')
        return tweet, hashtags, user

    def update_dataset_storage(self, dropdown_dataset):
        logger.debug(f'Updating dataset storage with {dropdown_dataset}')
        return dropdown_dataset

    def update_hashtag_storage(self, click_data):
        logger.debug(f'Updating hashtag storage with {click_data}')
        if click_data:
            hashtags = [click_data[0]]
            tweet_id = None
            user_id = None
            return hashtags, tweet_id, user_id
        else:
            raise PreventUpdate()

    def update_start_date_storage(self, start_date):
        logger.debug(f'Updating start date storage with {start_date}')
        return start_date

    def update_end_date_storage(self, end_date):
        logger.debug(f'Updating end date storage with {end_date}')
        return end_date

    def layout(self, params=None):
        return dbc.Stack([
            dbc.Card([
                dbc.CardHeader('Date range'),
                dbc.CardBody([
                    self.date_picker
                ])
            ]),
            self.filter_display.layout(),

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
            Output(self.state.current_tweet, 'data', allow_duplicate=True),
            Output(self.state.current_user, 'data', allow_duplicate=True),
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

        app.callback(
            Output(self.wordcloud, 'list'),
            Output(self.wordcloud, 'weightFactor'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data')],
        )(self.update_wordcloud)

        self.filter_display.callbacks(app)


class FilterDisplayComponent(RemissComponent):
    def __init__(self, state, name=None, gap=2):
        super().__init__(name)
        self.state = state
        self.tweet_display = html.P(id=f'tweet-display-{self.name}', children='')
        self.hashtags_display = html.P(id=f'hashtags-display-{self.name}', children='')
        self.user_display = html.P(id=f'user-display-{self.name}', children='')
        self.reset_button = dbc.Button('Clear filters', id=f'clear-filters-{self.name}', color='danger')
        self.tweet_card = self.create_field_card('collapse-tweet', 'Current Tweet:', self.tweet_display)
        self.user_card = self.create_field_card('collapse-user', 'Current User:', self.user_display)
        self.hashtags_card = self.create_field_card('collapse-hashtags', 'Current Hashtags:', self.hashtags_display)
        self.gap = gap

    def create_field_card(self, field_id, field_label, display):
        return dbc.Collapse([
            dbc.Card([
                dbc.CardHeader(field_label, id=f'header-{field_id}-{self.name}'),
                dbc.CardBody([
                    display
                ])]
            )], id=f'{field_id}-{self.name}', is_open=False)

    def layout(self, params=None):
        return dbc.Collapse([dbc.Card([
            dbc.CardHeader('Current Filters'),
            dbc.CardBody([
                dbc.Stack([
                    dbc.Row([
                        dbc.Col([
                            self.tweet_card
                        ]),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            self.user_card
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            self.hashtags_card
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            self.reset_button
                        ]),
                    ])
                ], gap=self.gap),
            ])
        ])], id=f'filter-display-{self.name}', is_open=False)

    def update_tweet_display(self, tweet):
        if tweet:
            return tweet, True
        else:
            return '', False

    def update_hashtags_display(self, hashtags):
        if hashtags:
            return ', '.join(hashtags), True
        else:
            return '', False

    def update_user_display(self, user):
        if user:
            return user, True
        else:
            return '', False

    def update_card_collapse(self, tweet, user, hashtags):
        return tweet or user or hashtags

    def clear_filters(self, n_clicks):
        if n_clicks:
            logger.debug(f'Clearing filters')
            return None, None, None
        raise PreventUpdate()

    def callbacks(self, app):
        app.callback(
            Output(self.tweet_display, 'children'),
            Output(self.tweet_card, 'is_open'),
            [Input(self.state.current_tweet, 'data')],
        )(self.update_tweet_display)

        app.callback(
            Output(self.hashtags_display, 'children'),
            Output(self.hashtags_card, 'is_open'),
            [Input(self.state.current_hashtags, 'data')],
        )(self.update_hashtags_display)

        app.callback(
            Output(self.user_display, 'children'),
            Output(self.user_card, 'is_open'),

            [Input(self.state.current_user, 'data')],
        )(self.update_user_display)

        app.callback(
            Output(f'filter-display-{self.name}', 'is_open'),
            [Input(self.state.current_tweet, 'data'),
             Input(self.state.current_user, 'data'),
             Input(self.state.current_hashtags, 'data')],
        )(self.update_card_collapse)

        app.callback(
            [Output(self.state.current_tweet, 'data', allow_duplicate=True),
             Output(self.state.current_hashtags, 'data', allow_duplicate=True),
             Output(self.state.current_user, 'data', allow_duplicate=True)],
            [Input(f'clear-filters-{self.name}', 'n_clicks')],
        )(self.clear_filters)
