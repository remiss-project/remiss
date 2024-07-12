from contextvars import copy_context
from unittest import TestCase

import dash_bootstrap_components as dbc
from dash import Dash, dcc, Output, Input
from dash._callback_context import context_value
from dash._utils import AttributeDict
from pymongo import MongoClient

from components.control_panel import ControlPanelComponent
from components.profiling import UserInfoComponent, TopicVerticalBarplotComponent, RadarplotEmotionsComponent, \
    VerticalAccumulatedBarplotAge, VerticalAccumulatedBarplotGenre, VerticalBarplotPolarityComponent, \
    HorizontalBarplotInteraction1, \
    HorizontalBarplotInteraction2, DonutPlotBehaviour1Component, DonutPlotBehaviour2Component, ProfilingComponent
from components.dashboard import RemissState
from figures import TimeSeriesFactory
from figures.profiling import ProfilingPlotFactory
from tests.conftest import populate_test_database, delete_test_database


class TestCVCComponents(TestCase):
    def setUp(self):
        self.propagation_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.test_user_id = '1033714286231740416'
        self.test_tweet_id = '1167078759280889856'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database(self.test_dataset)

    def test_plot_user_info_component(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = UserInfoComponent(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_user_info_component_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = UserInfoComponent(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_vertical_barplot_topics_component(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = TopicVerticalBarplotComponent(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_vertical_barplot_topics_component_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = TopicVerticalBarplotComponent(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_radarplot_emotions(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = RadarplotEmotionsComponent(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_radarplot_emotions_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = RadarplotEmotionsComponent(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_vertical_accumulated_barplot_age(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = VerticalAccumulatedBarplotAge(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 0

    def test_plot_vertical_accumulated_barplot_age_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = VerticalAccumulatedBarplotAge(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update()

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_vertical_accumulated_barplot_genre(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = VerticalAccumulatedBarplotGenre(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 0

    def test_plot_vertical_accumulated_barplot_genre_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = VerticalAccumulatedBarplotGenre(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update()

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_vertical_barplot_polarity(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = VerticalBarplotPolarityComponent(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_vertical_barplot_polarity_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = VerticalBarplotPolarityComponent(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_horizontal_bars_plot_interactions(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = HorizontalBarplotInteraction1(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_horizontal_bars_plot_interactions_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = HorizontalBarplotInteraction1(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_horizontal_bars_plot_interactions_2(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = HorizontalBarplotInteraction2(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_horizontal_bars_plot_interactions_2_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = HorizontalBarplotInteraction2(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_donut_plot_behaviour1(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = DonutPlotBehaviour1Component(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_donut_plot_behaviour1_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = DonutPlotBehaviour1Component(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def test_plot_donut_plot_behaviour2(self):
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        state = RemissState()
        component = DonutPlotBehaviour2Component(cvc_plot_factory, state)
        dash_app = Dash(__name__)
        component.callbacks(dash_app)

        assert len(dash_app.callback_map) == 1
        assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
        assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
        assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
            {'id': component.state.current_dataset.id, 'property': 'data'},
            {'id': component.state.current_user.id, 'property': 'data'}]
        assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

    def test_plot_donut_plot_behaviour2_run_callback(self):
        plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')
        time_series_factory = TimeSeriesFactory()
        state = RemissState()
        component = DonutPlotBehaviour2Component(plot_factory, state)
        control_panel = ControlPanelComponent(time_series_factory, state)

        dash_app = Dash(__name__)
        component.callbacks(dash_app)
        control_panel.callbacks(dash_app)
        dash_app.layout = dbc.Container([
            state.layout(),
            control_panel.layout(),
            component.layout(),
        ])

        def run_callback():
            context_value.set(AttributeDict({'inputs': {'current-dataset-state': 'data',
                                                        'current-start-date-state': 'data',
                                                        'current-end-date-state': 'data'}}))
            component.update(self.test_dataset, self.test_user_id)

        ctx = copy_context()
        output = ctx.run(run_callback)

    def _test_run_server(self):
        # create factories
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data')

        state = RemissState()
        # create components
        user_info_component = UserInfoComponent(cvc_plot_factory, state)
        topic_vertical_barplot_component = TopicVerticalBarplotComponent(cvc_plot_factory, state)
        radarplot_emotions_component = RadarplotEmotionsComponent(cvc_plot_factory, state)
        vertical_accumulated_barplot_age = VerticalAccumulatedBarplotAge(cvc_plot_factory, state)
        vertical_accumulated_barplot_genre = VerticalAccumulatedBarplotGenre(cvc_plot_factory, state)
        vertical_barplot_polarity = VerticalBarplotPolarityComponent(cvc_plot_factory, state)
        horizontal_barplot_interaction1 = HorizontalBarplotInteraction1(cvc_plot_factory, state)
        horizontal_barplot_interaction2 = HorizontalBarplotInteraction2(cvc_plot_factory, state)
        donut_plot_behaviour1 = DonutPlotBehaviour1Component(cvc_plot_factory, state)
        donut_plot_behaviour2 = DonutPlotBehaviour2Component(cvc_plot_factory, state)
        # create control panel
        # create dash app
        dash_app = Dash(__name__)
        # add callbacks
        user_info_component.callbacks(dash_app)
        topic_vertical_barplot_component.callbacks(dash_app)
        radarplot_emotions_component.callbacks(dash_app)
        vertical_accumulated_barplot_age.callbacks(dash_app)
        vertical_accumulated_barplot_genre.callbacks(dash_app)
        vertical_barplot_polarity.callbacks(dash_app)
        horizontal_barplot_interaction1.callbacks(dash_app)
        horizontal_barplot_interaction2.callbacks(dash_app)
        donut_plot_behaviour1.callbacks(dash_app)
        donut_plot_behaviour2.callbacks(dash_app)
        # create layout
        dash_app.layout = dbc.Container([
            state.layout(),
            user_info_component.layout(),
            topic_vertical_barplot_component.layout(),
            radarplot_emotions_component.layout(),
            vertical_accumulated_barplot_age.layout(),
            vertical_accumulated_barplot_genre.layout(),
            vertical_barplot_polarity.layout(),
            horizontal_barplot_interaction1.layout(),
            horizontal_barplot_interaction2.layout(),
            donut_plot_behaviour1.layout(),
            donut_plot_behaviour2.layout(),
        ])
        # run server
        dash_app.run(debug=True)

    def _test_render_profiling_component(self):
        # create factories
        cvc_plot_factory = ProfilingPlotFactory(data_dir='./../profiling_data', available_datasets=[self.test_dataset])

        state = RemissState()
        # create components
        profiling_component = ProfilingComponent(cvc_plot_factory, state)
        # create dash app
        dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
        dash_app = Dash(__name__,
                        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME, dbc_css],
                        prevent_initial_callbacks="initial_duplicate",
                        meta_tags=[
                            {
                                "name": "viewport",
                                "content": "width=device-width, initial-scale=1, maximum-scale=1",
                            }
                        ],
                        )
        # add callbacks
        profiling_component.callbacks(dash_app)
        # create layout
        dash_app.layout = dbc.Container([
            state.layout(),
            dbc.Row([
                dbc.Col([
                    profiling_component.layout(),
                ]),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Button('Toggle collapse', id='toggle-collapse')
                ])
            ]),
        ])

        def on_button_click(n):
            n = n if n is not None else 0
            user_id = None if n % 2 == 0 else self.test_user_id
            return self.test_dataset, user_id

        dash_app.callback(
            Output(profiling_component.state.current_dataset.id, 'data'),
            Output(profiling_component.state.current_user.id, 'data'),
            [Input('toggle-collapse', 'n_clicks')]
        )(on_button_click)

        dash_app.run(debug=True)
