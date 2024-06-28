import dash_bootstrap_components as dbc
from dash import dcc, Output, Input

from components.components import RemissComponent


class ProfilingComponent(RemissComponent):
    title = 'Profiling component'

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
                        self.graph
                    ])
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


class UserInfoComponent(ProfilingComponent):
    title = 'User info'

    def update(self, dataset, user):
        return self.plot_factory.plot_user_info(dataset, user)


class TopicVerticalBarplotComponent(ProfilingComponent):
    title = 'Topic vertical barplot'

    def update(self, dataset, user):
        return self.plot_factory.plot_vertical_barplot_topics(dataset, user)


class RadarplotEmotionsComponent(ProfilingComponent):
    title = 'Radarplot emotions'

    def update(self, dataset, user):
        return self.plot_factory.plot_radarplot_emotions(dataset, user)


class VerticalAccumulatedBarplotAge(ProfilingComponent):
    title = 'Vertical accumulated barplot by age'

    def update(self):
        return self.plot_factory.plot_vertical_accumulated_barplot_by_age()

    def callbacks(self, app):
        pass

# plot_vertical_accumulated_barplot_by_genre
class VerticalAccumulatedBarplotGenre(ProfilingComponent):
    title = 'Vertical accumulated barplot by genre'

    def update(self):
        return self.plot_factory.plot_vertical_accumulated_barplot_by_genre()

    def callbacks(self, app):
        pass

# plot_vertical_barplot_polarity
class VerticalBarplotPolarityComponent(ProfilingComponent):
    title = 'Vertical barplot polarity'

    def update(self, dataset, user):
        return self.plot_factory.plot_vertical_barplot_polarity(dataset, user)


# plot_horizontal_bars_plot_interactions

class HorizontalBarplotInteraction1(ProfilingComponent):
    title = 'Horizontal barplot interaction 1'

    def update(self, dataset, user):
        return self.plot_factory.plot_horizontal_bars_plot_interactions(dataset, user)[0]


class HorizontalBarplotInteraction2(ProfilingComponent):
    title = 'Horizontal barplot interaction 2'

    def update(self, dataset, user):
        return self.plot_factory.plot_horizontal_bars_plot_interactions(dataset, user)[1]


# plot_donut_plot_behaviour

class DonutPlotBehaviour1Component(ProfilingComponent):
    title = 'Donut plot behaviour 1'

    def update(self, dataset, user):
        return self.plot_factory.plot_donut_plot_behaviour(dataset, user)[0]


class DonutPlotBehaviour2Component(ProfilingComponent):
    title = 'Donut plot behaviour 2'

    def update(self, dataset, user):
        return self.plot_factory.plot_donut_plot_behaviour(dataset, user)[1]
