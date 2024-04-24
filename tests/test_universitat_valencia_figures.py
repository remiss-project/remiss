import json
from unittest.mock import Mock

import plotly
import pytest
from plotly.graph_objs import Figure
from pyvis.network import Network

from figures.universitat_valencia import UVAPIFactory


def test_fetch_plot_emotion_per_hour():
    plot_factory = UVAPIFactory()
    plotly_json = plot_factory.fetch_graph_json('graph1', 'madrid', None, None)
    # with open('test_resources/emotion_per_hour.json', 'w') as f:
    #     f.write(plotly_json)
    fig = plotly.io.from_json(plotly_json, skip_invalid=True)
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 14


def test_plot_emotion_per_hour():
    plot_factory = UVAPIFactory()
    with open('test_resources/emotion_per_hour.json', 'r') as f:
        plot_json = f.read()
    plot_factory.fetch_graph_json = Mock(return_value=plot_json)
    fig = plot_factory.plot_emotion_per_hour('madrid', 'start_time', 'end_time')
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 14


def test_fetch_plot_average_emotion():
    plot_factory = UVAPIFactory()
    plotly_json = plot_factory.fetch_graph_json('graph2', 'madrid', None, None)
    # with open('test_resources/average_emotion.json', 'w') as f:
    #     f.write(plotly_json)
    fig = plotly.io.from_json(plotly_json, skip_invalid=True)
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 11


def test_plot_average_emotion():
    plot_factory = UVAPIFactory()
    with open('test_resources/average_emotion.json', 'r') as f:
        plot_json = f.read()
    plot_factory.fetch_graph_json = Mock(return_value=plot_json)
    fig = plot_factory.plot_average_emotion('madrid', 'start_time', 'end_time')
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 11


def test_fetch_plot_top_profiles():
    plot_factory = UVAPIFactory()
    plotly_json = plot_factory.fetch_graph_json('graph3', 'madrid', None, None)
    with open('test_resources/top_profiles.json', 'w') as f:
        f.write(plotly_json)
    fig = plotly.io.from_json(plotly_json, skip_invalid=True)
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_plot_top_profiles():
    plot_factory = UVAPIFactory()
    with open('test_resources/top_profiles.json', 'r') as f:
        plot_json = f.read()
    plot_factory.fetch_graph_json = Mock(return_value=plot_json)
    fig = plot_factory.plot_top_profiles('madrid', 'start_time', 'end_time')
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_fetch_plot_top_hashtags():
    plot_factory = UVAPIFactory()
    plotly_json = plot_factory.fetch_graph_json('graph4', 'madrid', None, None)
    with open('test_resources/top_hashtags.json', 'w') as f:
        f.write(plotly_json)
    fig = plotly.io.from_json(plotly_json, skip_invalid=True)
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_plot_top_hashtags():
    plot_factory = UVAPIFactory()
    with open('test_resources/top_hashtags.json', 'r') as f:
        plot_json = f.read()
    plot_factory.fetch_graph_json = Mock(return_value=plot_json)
    fig = plot_factory.plot_top_hashtags('madrid', 'start_time', 'end_time')
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_fetch_plot_topic_ranking():
    plot_factory = UVAPIFactory()
    plotly_json = plot_factory.fetch_graph_json('graph5', 'madrid', None, None)
    with open('test_resources/topic_ranking.json', 'w') as f:
        f.write(plotly_json)
    fig = plotly.io.from_json(plotly_json, skip_invalid=True)
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


def test_plot_topic_ranking():
    plot_factory = UVAPIFactory()
    with open('test_resources/topic_ranking.json', 'r') as f:
        plot_json = f.read()
    plot_factory.fetch_graph_json = Mock(return_value=plot_json)
    fig = plot_factory.plot_topic_ranking('madrid', 'start_time', 'end_time')
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 2


@pytest.mark.skip(
    reason="Network retorna un graf visualitzat amb nosequin framework que no es plotly, aixi que es normal que falli")
def test_fetch_plot_network_topics():
    plot_factory = UVAPIFactory()
    visjs = plot_factory.fetch_graph_json('graph6', 'madrid', None, None)
    # with open('test_resources/network_topics.json', 'w') as f:
    #     f.write(visjs)

    network = Network()
    network.from_json(visjs)
    network.show()


@pytest.mark.skip(
    reason="Network retorna un graf visualitzat amb nosequin framework que no es plotly, aixi que es normal que falli")
def test_plot_network_topics():
    plot_factory = UVAPIFactory()
    with open('test_resources/network_topics.json', 'r') as f:
        plot_json = f.read()
    plot_factory.fetch_graph_json = Mock(return_value=plot_json)
    # fig = plot_factory.plot_network_topics('madrid', 'start_time', 'end_time')
    # fig.show()
    # assert isinstance(fig, Figure)
    # assert len(fig.data) > 1
    plot_json = json.loads(plot_json)
    network = Network(height='1000px', width='100%')
    for node in plot_json['Misnodes']:
        node['n_id'] = node.pop('id')
        network.add_node(**node)
    for edge in plot_json['Mislinks']:
        edge['source'] = edge.pop('from')
        network.add_edge(**edge)

    network.show('test_resources/network_topics_pyvis.html')


def test_plotly_json_to_figure():
    plot_factory = UVAPIFactory()
    with open('test_resources/emotion_per_hour.json', 'r') as f:
        plot_json = f.read()
    fig = plot_factory.plotly_json_to_figure(plot_json)
    fig.show()
    assert isinstance(fig, Figure)
    assert len(fig.data) == 14
