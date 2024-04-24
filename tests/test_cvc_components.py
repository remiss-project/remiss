from contextvars import copy_context

import dash_bootstrap_components as dbc
from dash import Dash
from dash._callback_context import context_value
from dash._utils import AttributeDict

from components.control_panel import ControlPanelComponent
from components.cvc import UserInfoComponent, TopicVerticalBarplotComponent, RadarplotEmotionsComponent, \
    VerticalAccumulatedBarplotAge, VerticalAccumulatedBarplotGenre, VerticalBarplotPolarity, \
    HorizontalBarplotInteraction1, \
    HorizontalBarplotInteraction2, DonutPlotBehaviour1, DonutPlotBehaviour2

from components.dashboard import RemissState
from figures import TimeSeriesFactory
from figures.cvc import CVCPlotFactory


def test_plot_user_info_component():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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


def test_plot_user_info_component_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)


def test_plot_vertical_barplot_topics_component():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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


def test_plot_vertical_barplot_topics_component_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)


def test_plot_radarplot_emotions():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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


def test_plot_radarplot_emotions_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)


def test_plot_vertical_accumulated_barplot_age():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    state = RemissState()
    component = VerticalAccumulatedBarplotAge(cvc_plot_factory, state)
    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 0


def test_plot_vertical_accumulated_barplot_age_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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


def test_plot_vertical_accumulated_barplot_genre():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    state = RemissState()
    component = VerticalAccumulatedBarplotGenre(cvc_plot_factory, state)
    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 0


def test_plot_vertical_accumulated_barplot_genre_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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


def test_plot_vertical_barplot_polarity():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    state = RemissState()
    component = VerticalBarplotPolarity(cvc_plot_factory, state)
    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_user.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


def test_plot_vertical_barplot_polarity_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = VerticalBarplotPolarity(plot_factory, state)
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)


def test_plot_horizontal_bars_plot_interactions():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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

def test_plot_horizontal_bars_plot_interactions_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)

def test_plot_horizontal_bars_plot_interactions_2():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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

def test_plot_horizontal_bars_plot_interactions_2_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)

def test_plot_donut_plot_behaviour1():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    state = RemissState()
    component = DonutPlotBehaviour1(cvc_plot_factory, state)
    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_user.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

def test_plot_donut_plot_behaviour1_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = DonutPlotBehaviour1(plot_factory, state)
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)

def test_plot_donut_plot_behaviour2():
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    state = RemissState()
    component = DonutPlotBehaviour2(cvc_plot_factory, state)
    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_user.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id

def test_plot_donut_plot_behaviour2_run_callback():
    plot_factory = CVCPlotFactory(data_dir='./../cvc_data')
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = DonutPlotBehaviour2(plot_factory, state)
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
        component.update('CVCFeatures2', '100485425')

    ctx = copy_context()
    output = ctx.run(run_callback)

def test_run_server():
    # create factories
    cvc_plot_factory = CVCPlotFactory(data_dir='./../cvc_data')

    state = RemissState()
    # create components
    user_info_component = UserInfoComponent(cvc_plot_factory, state)
    topic_vertical_barplot_component = TopicVerticalBarplotComponent(cvc_plot_factory, state)
    radarplot_emotions_component = RadarplotEmotionsComponent(cvc_plot_factory, state)
    vertical_accumulated_barplot_age = VerticalAccumulatedBarplotAge(cvc_plot_factory, state)
    vertical_accumulated_barplot_genre = VerticalAccumulatedBarplotGenre(cvc_plot_factory, state)
    vertical_barplot_polarity = VerticalBarplotPolarity(cvc_plot_factory, state)
    horizontal_barplot_interaction1 = HorizontalBarplotInteraction1(cvc_plot_factory, state)
    horizontal_barplot_interaction2 = HorizontalBarplotInteraction2(cvc_plot_factory, state)
    donut_plot_behaviour1 = DonutPlotBehaviour1(cvc_plot_factory, state)
    donut_plot_behaviour2 = DonutPlotBehaviour2(cvc_plot_factory, state)
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

