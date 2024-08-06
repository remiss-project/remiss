import json
import re

import plotly
import requests
from pymongo import MongoClient
import plotly.express as px

from figures.figures import MongoPlotFactory


class RemoteTextualFactory:
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


class TextualFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None):
        super().__init__(host, port, available_datasets)

    def plot_average_emotion(self, dataset, start_time, end_time):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        dataset = database.get_collection('textual')

        emotions = ['Miedo', 'Disgusto', 'Sorpresa', 'Odio', 'Diversion', 'Agresividad', 'Dirigido', 'Enfado',
                    'Tristeza', 'Toxico', 'Ironia']

        # Take average of each of the columns
        pipeline = [
            {'$group': {'_id': None, **{emotion: {'$avg': f'${emotion}'} for emotion in emotions}}},
            {'$project': {'_id': 0}}
        ]
        df = dataset.aggregate_pandas_all(pipeline).iloc[0]
        translations = {
            'Miedo': 'Fear',
            'Disgusto': 'Disgust',
            'Sorpresa': 'Surprise',
            'Odio': 'Hate',
            'Diversion': 'Fun',
            'Agresividad': 'Aggressiveness',
            'Dirigido': 'Targeted',
            'Enfado': 'Anger',
            'Tristeza': 'Sadness',
            'Toxico': 'Toxic',
            'Ironia': 'Irony'
        }
        df = df.rename(index=translations)
        fig = px.bar(x=df.index, y=df.values, labels={'x': 'Emotion', 'y': 'Average value'},
                     color=df.index)
        fig.update_layout(showlegend=False)
        return fig

    def get_emotion_per_hour(self, dataset, start_time, end_time):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        dataset = database.get_collection('textual')

        emotions = ['Agresividad', 'Enfado', 'Disgusto', 'Miedo', 'Odio', 'Ironia', 'Diversion', 'Tristeza', 'Sorpresa',
                    'Dirigido', 'Negativo', 'Neutro', 'Positivo']

        pipeline = [
            {'$addFields': {'date': {'$toDate': '$date'}}},
            {'$addFields': {'hour': {'$hour': '$date'}}},
            {'$group': {'_id': '$hour', **{emotion: {'$avg': f'${emotion}'} for emotion in emotions}}},
            {'$project': {'_id': 0}}
        ]
        # if start_time:
        #     pipeline.insert(2, {'$match': {'date': {'$gte': start_time}}})
        # if end_time:
        #     pipeline.insert(2, {'$match': {'date': {'$lte': end_time}}})
        df = dataset.aggregate_pandas_all(pipeline)
        return df

    def plot_emotion_per_hour(self, dataset, start_time, end_time):
        df = self.get_emotion_per_hour(dataset, start_time, end_time)
        translations = {
            'Agresividad': 'Aggressiveness',
            'Enfado': 'Anger',
            'Disgusto': 'Disgust',
            'Miedo': 'Fear',
            'Odio': 'Hate',
            'Ironia': 'Irony',
            'Diversion': 'Fun',
            'Tristeza': 'Sadness',
            'Sorpresa': 'Surprise',
            'Dirigido': 'Targeted',
            'Negativo': 'Negative',
            'Neutro': 'Neutral',
            'Positivo': 'Positive'
        }
        df = df.rename(columns=translations)

        fig = px.line(df, x=df.index, y=df.columns)
        return fig
