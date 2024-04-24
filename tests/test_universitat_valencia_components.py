from contextvars import copy_context
from unittest.mock import Mock

import dash_bootstrap_components as dbc
import pytest
from dash import Dash
from dash._callback_context import context_value
from dash._utils import AttributeDict

from components.control_panel import ControlPanelComponent
from components.dashboard import RemissState
from components.universitat_valencia import EmotionPerHourComponent, AverageEmotionBarComponent, TopProfilesComponent, \
    TopHashtagsComponent, TopicRankingComponent, NetworkTopicsComponent
from figures import TimeSeriesFactory
from figures.universitat_valencia import EmotionPerHourPlotFactory, AverageEmotionBarPlotFactory, \
    TopProfilesPlotFactory, TopHashtagsPlotFactory, TopicRankingPlotFactory, NetworkTopicsPlotFactory


def test_emotion_per_hour_component():
    plot_factory = EmotionPerHourPlotFactory()
    state = RemissState()
    component = EmotionPerHourComponent(plot_factory, state)

    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_start_date.id, 'property': 'data'},
        {'id': component.state.current_end_date.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


def test_emotion_per_hour_component_run_server():
    with open('test_resources/emotion_per_hour.html', 'r') as f:
        expected = f.read()

    plot_factory = EmotionPerHourPlotFactory()
    plot_factory.fetch_graph_json = Mock(return_value=expected)
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = EmotionPerHourComponent(plot_factory, state)
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
        component.update('madrid', 'start_time', 'end_time')

    ctx = copy_context()
    output = ctx.run(run_callback)
    print(output)
    plot_factory.fetch_graph_json.assert_called_with('madrid', 'start_time', 'end_time')


def test_average_emotion_bar_component():
    plot_factory = AverageEmotionBarPlotFactory()
    state = RemissState()
    component = AverageEmotionBarComponent(plot_factory, state)

    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_start_date.id, 'property': 'data'},
        {'id': component.state.current_end_date.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


def test_average_emotion_bar_component_run_server():
    with open('test_resources/average_emotion.html', 'r') as f:
        expected = f.read()

    plot_factory = AverageEmotionBarPlotFactory()
    plot_factory.fetch_graph_json = Mock(return_value=expected)
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = AverageEmotionBarComponent(plot_factory, state)
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
        component.update('madrid', 'start_time', 'end_time')

    ctx = copy_context()
    output = ctx.run(run_callback)
    print(output)
    plot_factory.fetch_graph_json.assert_called_with('madrid', 'start_time', 'end_time')


def test_top_profiles_component():
    plot_factory = TopProfilesPlotFactory()
    state = RemissState()
    component = TopProfilesComponent(plot_factory, state)

    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_start_date.id, 'property': 'data'},
        {'id': component.state.current_end_date.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


def test_top_profiles_component_run_server():
    with open('test_resources/top_profiles.html', 'r') as f:
        expected = f.read()

    plot_factory = TopProfilesPlotFactory()
    plot_factory.fetch_graph_json = Mock(return_value=expected)
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = TopProfilesComponent(plot_factory, state)
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
        component.update('madrid', 'start_time', 'end_time')

    ctx = copy_context()
    output = ctx.run(run_callback)
    print(output)
    plot_factory.fetch_graph_json.assert_called_with('madrid', 'start_time', 'end_time')


def test_top_hashtags_component():
    plot_factory = TopHashtagsPlotFactory()
    state = RemissState()
    component = TopHashtagsComponent(plot_factory, state)

    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_start_date.id, 'property': 'data'},
        {'id': component.state.current_end_date.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


def test_top_hashtags_component_run_server():
    with open('test_resources/top_hashtags.html', 'r') as f:
        expected = f.read()

    plot_factory = TopHashtagsPlotFactory()
    plot_factory.fetch_graph_json = Mock(return_value=expected)
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = TopHashtagsComponent(plot_factory, state)
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
        component.update('madrid', 'start_time', 'end_time')

    ctx = copy_context()
    output = ctx.run(run_callback)
    print(output)
    plot_factory.fetch_graph_json.assert_called_with('madrid', 'start_time', 'end_time')


def test_topic_ranking_component():
    plot_factory = TopicRankingPlotFactory()
    state = RemissState()
    component = TopicRankingComponent(plot_factory, state)

    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_start_date.id, 'property': 'data'},
        {'id': component.state.current_end_date.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


def test_topic_ranking_component_run_server():
    with open('test_resources/topic_ranking.html', 'r') as f:
        expected = f.read()

    plot_factory = TopicRankingPlotFactory()
    plot_factory.fetch_graph_json = Mock(return_value=expected)
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = TopicRankingComponent(plot_factory, state)
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
        component.update('madrid', 'start_time', 'end_time')

    ctx = copy_context()
    output = ctx.run(run_callback)
    print(output)
    plot_factory.fetch_graph_json.assert_called_with('madrid', 'start_time', 'end_time')


def test_network_topics_component():
    plot_factory = NetworkTopicsPlotFactory()
    state = RemissState()
    component = NetworkTopicsComponent(plot_factory, state)

    dash_app = Dash(__name__)
    component.callbacks(dash_app)

    assert len(dash_app.callback_map) == 1
    assert list(dash_app.callback_map.keys()) == [component.graph.id + '.figure']
    assert dash_app.callback_map[component.graph.id + '.figure']['callback'].__name__ == component.update.__name__
    assert dash_app.callback_map[component.graph.id + '.figure']['inputs'] == [
        {'id': component.state.current_dataset.id, 'property': 'data'},
        {'id': component.state.current_start_date.id, 'property': 'data'},
        {'id': component.state.current_end_date.id, 'property': 'data'}]
    assert dash_app.callback_map[component.graph.id + '.figure']['output'].component_id == component.graph.id


@pytest.mark.skip(
    reason="Network retorna un graf visualitzat amb nosequin framework que no es plotly, aixi que es normal que falli")
def test_network_topics_component_run_server():
    with open('test_resources/network_topics.html', 'r') as f:
        expected = f.read()

    plot_factory = NetworkTopicsPlotFactory()
    plot_factory.fetch_graph_json = Mock(return_value=expected)
    time_series_factory = TimeSeriesFactory()
    state = RemissState()
    component = NetworkTopicsComponent(plot_factory, state)
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
        component.update('madrid', 'start_time', 'end_time')

    ctx = copy_context()
    output = ctx.run(run_callback)
    print(output)
    plot_factory.fetch_graph_json.assert_called_with('madrid', 'start_time', 'end_time')
