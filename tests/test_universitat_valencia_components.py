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
from figures.universitat_valencia import UVAPIFactory


def test_emotion_per_hour_component():
    plot_factory = UVAPIFactory()
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
    with open('test_resources/emotion_per_hour.json', 'r') as f:
        expected = f.read()

    plot_factory = UVAPIFactory()
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
    plot_factory.fetch_graph_json.assert_called_with('graph1', 'madrid', 'start_time', 'end_time')


def test_average_emotion_bar_component():
    plot_factory = UVAPIFactory()
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
    with open('test_resources/average_emotion.json', 'r') as f:
        expected = f.read()

    plot_factory = UVAPIFactory()
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
    plot_factory.fetch_graph_json.assert_called_with('graph2', 'madrid', 'start_time', 'end_time')


def test_top_profiles_component():
    plot_factory = UVAPIFactory()
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
    with open('test_resources/top_profiles.json', 'r') as f:
        expected = f.read()

    plot_factory = UVAPIFactory()
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
    plot_factory.fetch_graph_json.assert_called_with('graph3', 'madrid', 'start_time', 'end_time')


def test_top_hashtags_component():
    plot_factory = UVAPIFactory()
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
    with open('test_resources/top_hashtags.json', 'r') as f:
        expected = f.read()

    plot_factory = UVAPIFactory()
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
    plot_factory.fetch_graph_json.assert_called_with('graph4', 'madrid', 'start_time', 'end_time')


def test_topic_ranking_component():
    plot_factory = UVAPIFactory()
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
    with open('test_resources/topic_ranking.json', 'r') as f:
        expected = f.read()

    plot_factory = UVAPIFactory()
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
    plot_factory.fetch_graph_json.assert_called_with('graph5', 'madrid', 'start_time', 'end_time')


def test_network_topics_component():
    plot_factory = UVAPIFactory()
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
    with open('test_resources/network_topics.json', 'r') as f:
        expected = f.read()

    plot_factory = UVAPIFactory()
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


def test_uv_demo():
    # create factory
    state = RemissState()
    plot_factory = UVAPIFactory()
    # mock api calls so it does not take ages
    def fetch_graph_json(graph_id, dataset, start_time=None, end_time=None):
        graphs = ['emotion_per_hour', 'average_emotion', 'top_profiles', 'top_hashtags', 'topic_ranking', 'network_topics']
        graph = graphs[int(graph_id[-1]) - 1]
        with open(f'test_resources/{graph}.json', 'r') as f:
            return f.read()
    plot_factory.fetch_graph_json = fetch_graph_json
    # create components
    emotion_per_hour = EmotionPerHourComponent(plot_factory, state)
    average_emotion = AverageEmotionBarComponent(plot_factory, state)
    top_profiles = TopProfilesComponent(plot_factory, state)
    top_hashtags = TopHashtagsComponent(plot_factory, state)
    topic_ranking = TopicRankingComponent(plot_factory, state)
    # network_topics = NetworkTopicsComponent(plot_factory, state)
    # create control panel
    time_series_factory = TimeSeriesFactory()
    control_panel = ControlPanelComponent(time_series_factory, state)
    # create dash app
    dash_app = Dash(__name__)
    # add callbacks
    emotion_per_hour.callbacks(dash_app)
    average_emotion.callbacks(dash_app)
    top_profiles.callbacks(dash_app)
    top_hashtags.callbacks(dash_app)
    topic_ranking.callbacks(dash_app)
    # network_topics.callbacks(dash_app)
    control_panel.callbacks(dash_app)
    # create layout
    dash_app.layout = dbc.Container([
        state.layout(),
        control_panel.layout(),
        emotion_per_hour.layout(),
        average_emotion.layout(),
        top_profiles.layout(),
        top_hashtags.layout(),
        topic_ranking.layout(),
        # network_topics.layout(),
    ])
    # run server
    dash_app.run(debug=True)
