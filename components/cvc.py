import dash_bootstrap_components as dbc
from dash import dcc, Output, Input

from components.components import RemissComponent


class CVCComponent(RemissComponent):
    title = 'CVC'

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

    def update(self, dataset, user):
        raise NotImplementedError()

    def callbacks(self, app):
        app.callback(
            Output(self.graph, 'figure'),
            [Input(self.state.current_dataset, 'data'),
             Input(self.state.current_user, 'data')],
        )(self.update)


class UserInfoComponent(CVCComponent):
    title = 'User info'

    def update(self, dataset, user):
        return self.plot_factory.plot_user_info(dataset, user)


class TopicVerticalBarplotComponent(CVCComponent):
    title = 'Topic vertical barplot'

    def update(self, dataset, user):
        return self.plot_factory.plot_topic_vertical_barplot(dataset, user)


class RadarplotEmotionsComponent(CVCComponent):
    title = 'Radarplot emotions'

    def update(self, dataset, user):
        return self.plot_factory.plot_radarplot_emotions(dataset, user)


class VerticalAccumulatedBarplotAge(CVCComponent):
    title = 'Vertical accumulated barplot by age'

    def update(self, dataset, user):
        return self.plot_factory.plot_vertical_accumulated_barplot_by_age(dataset, user)


# plot_vertical_barplot_polarity
class VerticalBarplotPolarity(CVCComponent):
    title = 'Vertical barplot polarity'

    def update(self, dataset, user):
        return self.plot_factory.plot_vertical_barplot_polarity(dataset, user)


# plot_horizontal_bars_plot_interactions

class HorizontalBarplotInteraction1(CVCComponent):
    title = 'Horizontal barplot interaction 1'

    def update(self, dataset, user):
        return self.plot_factory.plot_horizontal_bars_plot_interactions(dataset, user)[0]


class HorizontalBarplotInteraction2(CVCComponent):
    title = 'Horizontal barplot interaction 2'

    def update(self, dataset, user):
        return self.plot_factory.plot_horizontal_bars_plot_interactions(dataset, user)[1]


# plot_donut_plot_behaviour

class DonutPlotBehaviour(CVCComponent):
    title = 'Donut plot behaviour 1'

    def update(self, dataset, user):
        return self.plot_factory.plot_donut_plot_behaviour(dataset, user)[0]


class DonutPlotBehaviour2(CVCComponent):
    title = 'Donut plot behaviour 2'

    def update(self, dataset, user):
        return self.plot_factory.plot_donut_plot_behaviour(dataset, user)[1]
