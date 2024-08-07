import logging

import pandas as pd
from pymongo import MongoClient

from propagation import DiffusionMetrics, NetworkMetrics, Egonet

logger = logging.getLogger(__name__)


class PropagationPreprocessor:
    def __init__(self, dataset, data, host='localhost', port=27017, reference_types=('retweeted', 'quoted')):
        self.dataset = dataset
        self.data = data
        cast_strings_to_timestamps(self.data)
        self.host = host
        self.port = port
        self.reference_types = reference_types

        self.diffusion_metrics = DiffusionMetrics(host=self.host, port=self.port, reference_types=self.reference_types)
        self.network_metrics = NetworkMetrics(host=self.host, port=self.port, reference_types=self.reference_types)
        self.egonet = Egonet(host=self.host, port=self.port, reference_types=self.reference_types)

    def process(self):
        logger.info(f"Processing dataset {self.dataset} with data {self.data}")
        # Send data to raw collection
        logger.info(f"Storing data in raw collection")
        try:
            self.store_raw_data()
        except Exception as e:
            logger.error(f"Error storing data in raw collection: {e}")
            raise RuntimeError(f"Error storing data in raw collection: {e}") from e
        logger.info('Generating diffusion metrics')
        try:
            self.diffusion_metrics.persist([self.dataset])
        except Exception as e:
            logger.error(f"Error generating diffusion metrics: {e}")
            raise RuntimeError(f"Error generating diffusion metrics: {e}") from e
        logger.info('Generating network metrics')
        try:
            self.network_metrics.persist([self.dataset])
        except Exception as e:
            logger.error(f"Error generating network metrics: {e}")
            raise RuntimeError(f"Error generating network metrics: {e}") from e
        logger.info('Generating egonet metrics')
        try:
            self.egonet.persist([self.dataset])
        except Exception as e:
            logger.error(f"Error generating egonet metrics: {e}")
            raise RuntimeError(f"Error generating egonet metrics: {e}") from e

    def store_raw_data(self):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        database.drop_collection('raw')
        collection = database.get_collection('raw')
        collection.insert_many(self.data)
        logger.info('Data stored in raw collection')


def cast_strings_to_timestamps(data):
    for tweet in data:
        cast_tweet_strings_to_timestamps(tweet)


def cast_tweet_strings_to_timestamps(tweet):
    date_fields = {'created_at', 'editable_until', 'retrieved_at'}
    for field, value in tweet.items():
        if field in date_fields:
            if not isinstance(value, pd.Timestamp):
                tweet[field] = pd.Timestamp(value)

        elif isinstance(value, dict):
            cast_tweet_strings_to_timestamps(value)
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, dict):
                    cast_tweet_strings_to_timestamps(v)
