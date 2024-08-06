import logging

import dash_bootstrap_components as dbc
from dash import dcc, Output, Input

from components.components import RemissComponent

logger = logging.getLogger(__name__)


class ProfilingComponent(RemissComponent):

    def __init__(self, profile_plot_factory, state, name=None):
        super().__init__(name=name)
        self.profile_plot_factory = profile_plot_factory
        self.state = state
        self.radarplot_emotions = RadarplotEmotionsComponent(profile_plot_factory, state,
                                                             name=f'radarplot-emotions-{self.name}')
        self.vertical_barplot_polarity = VerticalBarplotPolarityComponent(profile_plot_factory, state,
                                                                          name=f'vertical-barplot-polarity-{self.name}')
        self.donut_plot_behaviour1 = DonutPlotBehaviour1Component(profile_plot_factory, state,
                                                                  name=f'donut-plot-behaviour1-{self.name}')
        self.donut_plot_behaviour2 = DonutPlotBehaviour2Component(profile_plot_factory, state,
                                                                  name=f'donut-plot-behaviour2-{self.name}')

    def layout(self, params=None):
        return dbc.Collapse([
            self.radarplot_emotions.layout(),
            self.vertical_barplot_polarity.layout(),
            self.donut_plot_behaviour1.layout(),
            self.donut_plot_behaviour2.layout(),
        ], id=f'collapse-{self.name}', is_open=False)

    def update_collapse(self, dataset, user_id):
        logger.info(f'Updating collapse with dataset {dataset}, user {user_id}')
        return self.profile_plot_factory.is_user_profiled(dataset, user_id)

    def callbacks(self, app):
        self.radarplot_emotions.callbacks(app)
        self.vertical_barplot_polarity.callbacks(app)
        self.donut_plot_behaviour1.callbacks(app)
        self.donut_plot_behaviour2.callbacks(app)
        app.callback(
            Output(f'collapse-{self.name}', 'is_open'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data')]
        )(self.update_collapse)


class BaseProfilingComponent(RemissComponent):
    title = 'Profiling component'
    caption = 'Profiling caption'

    def __init__(self, plot_factory, state, name=None):
        super().__init__(name=name)
        self.plot_factory = plot_factory
        self.graph = dcc.Graph(figure={}, id=f'fig-{self.name}')
        self.state = state

    def layout(self, params=None):
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(self.title),
                    dbc.CardBody([
                        dcc.Loading(
                            id=f'loading-{self.name}',
                            type='default',
                            children=self.graph
                        )
                    ]),
                    dbc.CardFooter(self.caption)
                ]),
            ]),
        ], justify='center', style={'margin-bottom': '1rem'})

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data')],
        )(self.update)


class UserInfoComponent(BaseProfilingComponent):
    title = 'User info'
    caption = 'General user information'

    def update(self, dataset, user):
        return self.plot_factory.plot_user_info(dataset, user)


class TopicVerticalBarplotComponent(BaseProfilingComponent):
    title = 'Topic vertical barplot'

    def update(self, dataset, user):
        return self.plot_factory.plot_vertical_barplot_topics(dataset, user)


class RadarplotEmotionsComponent(BaseProfilingComponent):
    title = 'Radarplot emotions'
    caption = 'Radarplot showing the distribution of emotions of the tweets produced by the user'

    def update(self, dataset, user):
        try:
            return self.plot_factory.plot_radarplot_emotions(dataset, user)
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            return {}


class VerticalAccumulatedBarplotAge(BaseProfilingComponent):
    title = 'Vertical accumulated barplot by age'

    def update(self):
        return self.plot_factory.plot_vertical_accumulated_barplot_by_age()

    def callbacks(self, app):
        pass


# plot_vertical_accumulated_barplot_by_genre
class VerticalAccumulatedBarplotGenre(BaseProfilingComponent):
    title = 'Vertical accumulated barplot by genre'

    def update(self):
        return self.plot_factory.plot_vertical_accumulated_barplot_by_genre()

    def callbacks(self, app):
        pass


# plot_vertical_barplot_polarity
class VerticalBarplotPolarityComponent(BaseProfilingComponent):
    title = 'Vertical barplot polarity'
    caption = 'Vertical barplot showing the distribution of polarity of the tweets produced by the user'

    def update(self, dataset, user):
        try:
            return self.plot_factory.plot_vertical_barplot_polarity(dataset, user)
        except TypeError as e:
            return {}


# plot_horizontal_bars_plot_interactions

class HorizontalBarplotInteraction1(BaseProfilingComponent):
    title = 'Horizontal barplot interaction 1'

    def update(self, dataset, user):
        return self.plot_factory.plot_horizontal_bars_plot_interactions(dataset, user)[0]


class HorizontalBarplotInteraction2(BaseProfilingComponent):
    title = 'Horizontal barplot interaction 2'

    def update(self, dataset, user):
        return self.plot_factory.plot_horizontal_bars_plot_interactions(dataset, user)[1]


# plot_donut_plot_behaviour

class DonutPlotBehaviour1Component(BaseProfilingComponent):
    title = 'Donut plot behaviour 1'
    caption = 'Donut plot showing the distribution of the days of the week of the tweets produced by the user ' \
              'compared to other types of users'

    def update(self, dataset, user):
        return self.plot_factory.plot_donut_plot_behaviour(dataset, user)[0]


class DonutPlotBehaviour2Component(BaseProfilingComponent):
    title = 'Donut plot behaviour 2'
    caption = 'Donut plot showing the distribution of the hours of the day of the tweets produced by the user ' \
              'compared to other types of users'

    def update(self, dataset, user):
        return self.plot_factory.plot_donut_plot_behaviour(dataset, user)[1]
