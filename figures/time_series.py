import logging
import time

import plotly.express as px
from propagation.histogram import Histogram
from pymongoarrow.monkey import patch_all

from figures.figures import MongoPlotFactory

patch_all()

logger = logging.getLogger('time_series')


class TimeSeriesFactory(MongoPlotFactory):

    def __init__(self, host='localhost', port=27017, available_datasets=None):
        super().__init__(host, port, available_datasets)
        self.histogram = Histogram(host=host, port=port)

    def plot_tweet_series(self, dataset, hashtags, start_time, end_time, unit='day', bin_size=1):
        start_time, end_time = self._validate_dates(dataset, start_time, end_time)
        if hashtags or start_time or end_time:
            try:
                logger.debug('Computing tweet series')
                start_computing_time = time.time()
                df = self.histogram.compute_tweet_histogram(dataset, hashtags, start_time, end_time, unit, bin_size)
                logger.debug(f'Tweet series computed in {time.time() - start_computing_time} seconds')
            except Exception as e:
                logger.error(f'Error computing tweet series: {e}')
                raise RuntimeError(f'Error computing tweet series: {e}') from e
        else:
            try:
                logger.debug('Loading tweet series from database')
                df = self.histogram.load_histogram(dataset, 'tweet')
            except Exception as e:
                logger.error(f'Error loading tweet series: {e}')
                raise RuntimeError('Error loading tweet series') from e

        plot = self._get_count_plot(df)
        return plot

    def plot_user_series(self, dataset, hashtags, start_time, end_time, unit='day', bin_size=1):
        start_time, end_time = self._validate_dates(dataset, start_time, end_time)
        if hashtags or start_time or end_time:
            try:
                logger.debug('Computing user series')
                start_computing_time = time.time()
                df = self.histogram.compute_user_histogram(dataset, hashtags, start_time, end_time, unit, bin_size)
                logger.debug(f'User series computed in {time.time() - start_computing_time} seconds')
            except Exception as e:
                logger.error(f'Error computing user series: {e}')
                raise RuntimeError('Error computing user series') from e
        else:
            try:
                logger.debug('Loading user series from database')
                df = self.histogram.load_histogram(dataset, 'user')
            except Exception as e:
                logger.error(f'Error loading user series: {e}')
                raise RuntimeError('Error loading user series') from e

        plot = self._get_count_plot(df)
        return plot

    def _get_count_plot(self, df):
        if len(df) == 1:
            plot = px.bar(df, labels={"value": "Count"})
        else:
            plot = px.area(df, labels={"value": "Count"}, )

        # Set all but the first trace to legendonly
        for i in range(1, len(plot.data)):
            plot.data[i]['visible'] = 'legendonly'

        return plot
