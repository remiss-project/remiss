from html.parser import HTMLParser
from unittest.mock import Mock

from plotly.graph_objs import Figure

from figures.universitat_valencia import EmotionPerHourPlotFactory, AverageEmotionBarPlotFactory, \
    TopProfilesPlotFactory, TopHashtagsPlotFactory, TopicRankingPlotFactory, NetworkTopicsPlotFactory


def test_fetch_plot_html_emotion_per_hour():
    plot_factory = EmotionPerHourPlotFactory()
    html = plot_factory.fetch_plot_html('madrid', 'start_time', 'end_time')
    # Validate that html is actually a valid html with some plotly stuff in it
    parser = HTMLParser()
    parser.feed(html)
    with open('test_resources/emotion_per_hour.html', 'w') as f:
        f.write(html)
    assert 'plotly' in html
    assert 'newPlot' in html


def test_plot_emotion_line_per_hour():
    plot_factory = EmotionPerHourPlotFactory()
    with open('test_resources/emotion_per_hour.html', 'r') as f:
        html = f.read()
    plot_factory.fetch_plot_html = Mock(return_value=html)
    fig = plot_factory.plot_emotion_per_hour('madrid', 'start_time', 'end_time')
    assert isinstance(fig, Figure)


def test_plotly_html_to_figure_emotion_per_hour():
    plot_factory = EmotionPerHourPlotFactory()
    with open('test_resources/emotion_per_hour.html', 'r') as f:
        html = f.read()
    fig = plot_factory.plotly_html_to_figure(html)
    assert isinstance(fig, Figure)


def test_fetch_plot_html_average_emotion():
    plot_factory = AverageEmotionBarPlotFactory()
    html = plot_factory.fetch_plot_html('madrid', 'start_time', 'end_time')
    # Validate that html is actually a valid html with some plotly stuff in it
    parser = HTMLParser()
    parser.feed(html)
    # with open('test_resources/average_emotion.html', 'w') as f:
    #     f.write(html)
    assert 'plotly' in html
    assert 'newPlot' in html


def test_plot_average_emotion_bar():
    plot_factory = AverageEmotionBarPlotFactory()
    with open('test_resources/average_emotion.html', 'r') as f:
        html = f.read()
    plot_factory.fetch_plot_html = Mock(return_value=html)
    fig = plot_factory.plot_average_emotion('madrid', 'start_time', 'end_time')
    assert isinstance(fig, Figure)


def test_plotly_html_to_figure_average_emotion():
    plot_factory = AverageEmotionBarPlotFactory()
    with open('test_resources/average_emotion.html', 'r') as f:
        html = f.read()
    fig = plot_factory.plotly_html_to_figure(html)
    assert isinstance(fig, Figure)


def test_fetch_plot_html_top_profiles():
    plot_factory = TopProfilesPlotFactory()
    html = plot_factory.fetch_plot_html('madrid', 'start_time', 'end_time')
    # Validate that html is actually a valid html with some plotly stuff in it
    parser = HTMLParser()
    parser.feed(html)
    # with open('test_resources/top_profiles.html', 'w') as f:
    #     f.write(html)
    assert 'plotly' in html
    assert 'newPlot' in html


def test_plot_top_profiles():
    plot_factory = TopProfilesPlotFactory()
    with open('test_resources/top_profiles.html', 'r') as f:
        html = f.read()
    plot_factory.fetch_plot_html = Mock(return_value=html)
    fig = plot_factory.plot_top_profiles('madrid', 'start_time', 'end_time')
    assert isinstance(fig, Figure)


def test_plotly_html_to_figure_top_profiles():
    plot_factory = TopProfilesPlotFactory()
    with open('test_resources/top_profiles.html', 'r') as f:
        html = f.read()
    fig = plot_factory.plotly_html_to_figure(html)
    assert isinstance(fig, Figure)


def test_fetch_plot_html_top_hashtags():
    plot_factory = TopHashtagsPlotFactory()
    html = plot_factory.fetch_plot_html('madrid', 'start_time', 'end_time')
    # Validate that html is actually a valid html with some plotly stuff in it
    parser = HTMLParser()
    parser.feed(html)
    # with open('test_resources/top_hashtags.html', 'w') as f:
    #     f.write(html)
    assert 'plotly' in html
    assert 'newPlot' in html


def test_plot_top_hashtags():
    plot_factory = TopHashtagsPlotFactory()
    with open('test_resources/top_hashtags.html', 'r') as f:
        html = f.read()
    plot_factory.fetch_plot_html = Mock(return_value=html)
    fig = plot_factory.plot_top_hashtags('madrid', 'start_time', 'end_time')
    assert isinstance(fig, Figure)


def test_plotly_html_to_figure_top_hashtags():
    plot_factory = TopHashtagsPlotFactory()
    with open('test_resources/top_hashtags.html', 'r') as f:
        html = f.read()
    fig = plot_factory.plotly_html_to_figure(html)
    assert isinstance(fig, Figure)


def test_fetch_plot_html_topic_ranking():
    plot_factory = TopicRankingPlotFactory()
    html = plot_factory.fetch_plot_html('castleon', 'start_time', 'end_time')
    # Validate that html is actually a valid html with some plotly stuff in it
    parser = HTMLParser()
    parser.feed(html)
    # with open('test_resources/topic_ranking.html', 'w') as f:
    #     f.write(html)
    assert 'plotly' in html
    assert 'newPlot' in html


def test_plot_topic_ranking():
    plot_factory = TopicRankingPlotFactory()
    with open('test_resources/topic_ranking.html', 'r') as f:
        html = f.read()
    plot_factory.fetch_plot_html = Mock(return_value=html)
    fig = plot_factory.plot_topic_ranking('castleon', 'start_time', 'end_time')
    assert isinstance(fig, Figure)


def test_plotly_html_to_figure_topic_ranking():
    plot_factory = TopicRankingPlotFactory()
    with open('test_resources/topic_ranking.html', 'r') as f:
        html = f.read()
    fig = plot_factory.plotly_html_to_figure(html)
    assert isinstance(fig, Figure)


def test_fetch_plot_html_network_topics():
    plot_factory = NetworkTopicsPlotFactory()
    html = plot_factory.fetch_plot_html('castleon', 'start_time', 'end_time')
    # Validate that html is actually a valid html with some plotly stuff in it
    parser = HTMLParser()
    parser.feed(html)
    with open('test_resources/network_topics.html', 'w') as f:
        f.write(html)
    assert 'plotly' in html
    assert 'newPlot' in html


def test_plot_network_topics():
    plot_factory = NetworkTopicsPlotFactory()
    with open('test_resources/network_topics.html', 'r') as f:
        html = f.read()
    plot_factory.fetch_plot_html = Mock(return_value=html)
    fig = plot_factory.plot_network_topics('castleon', 'start_time', 'end_time')
    assert isinstance(fig, Figure)

def test_plotly_html_to_figure_network_topics():
    plot_factory = NetworkTopicsPlotFactory()
    with open('test_resources/network_topics.html', 'r') as f:
        html = f.read()
    fig = plot_factory.plotly_html_to_figure(html)
    assert isinstance(fig, Figure)