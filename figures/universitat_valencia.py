import requests

from figures.figures import RemoteAPIFactory


class UVChart1PlotFactory(RemoteAPIFactory):
    def __init__(self, host, port, database, url, available_datasets=None):
        super().__init__(host, port, database, available_datasets)
        self.url = url

    def plot_user_sentiment(self, user_id, start_time, end_time):
        response = requests.get(f'{self.url}/user_sentiment/{user_id}', params={'start_time': start_time, 'end_time': end_time})
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f'Failed to get user sentiment for user {user_id}: {response.text}')

