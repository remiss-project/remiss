import json
import re

import plotly
import requests


class TextualFactory:
    def __init__(self, api_url='http://srvinv02.esade.es:5005/api'):
        super().__init__()
        self.api_url = api_url

    def plotly_json_to_figure(self, plotly_json):
        return plotly.io.from_json(plotly_json, skip_invalid=True)

    def plotly_html_to_figure(self, html):
        data_str = re.findall(r'<script type="application/json" data-for="htmlwidget-.*">(.*)</script>', html)[-1]
        call_args = json.loads(f'[{data_str}]')
        data = call_args[0]['x']['data']
        layout = call_args[0]['x']['layout']
        plotly_json = {'data': data, 'layout': layout}

        return plotly.io.from_json(json.dumps(plotly_json), skip_invalid=True)

    def fetch_graph_json(self, graph_id, dataset, start_time=None, end_time=None):
        response = requests.get(f'{self.api_url}/{graph_id}',
                                params={'name': dataset, 'start_time': start_time, 'end_time': end_time})
        if response.status_code == 200:
            # Return plotly figure
            return response.text

        else:
            raise RuntimeError(f'Error {response.status_code} while fetching {graph_id}.')

    def plot_emotion_per_hour(self, dataset, start_time, end_time):
        return self.plotly_json_to_figure(self.fetch_graph_json('graph1', dataset, start_time, end_time))

    def plot_average_emotion(self, dataset, start_time, end_time):
        return self.plotly_json_to_figure(self.fetch_graph_json('graph2', dataset, start_time, end_time))

    def plot_top_profiles(self, dataset, start_time, end_time):
        return self.plotly_json_to_figure(self.fetch_graph_json('graph3', dataset, start_time, end_time))

    def plot_top_hashtags(self, dataset, start_time, end_time):
        return self.plotly_json_to_figure(self.fetch_graph_json('graph4', dataset, start_time, end_time))

    def plot_topic_ranking(self, dataset, start_time, end_time):
        return self.plotly_json_to_figure(self.fetch_graph_json('graph5', dataset, start_time, end_time))

    def plot_network_topics(self, dataset, start_time, end_time):
        return self.visjs_json_to_figure(self.fetch_graph_json('graph6', dataset, start_time, end_time))

    def visjs_json_to_figure(self, visjs_json):
        raise NotImplementedError('Method not implemented yet')
