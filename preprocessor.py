import logging

import pandas as pd
from pymongo import MongoClient

from prepopulate import Prepopulator

logger = logging.getLogger(__name__)


class PropagationPreprocessor:
    def __init__(self, dataset, data, host='localhost', port=27017,
                 reference_types=('retweeted', 'quoted', 'replied_to'),
                 graph_layout='fruchterman_reingold',
                 propagation_threshold=0.2, propagation_frequency='1D', max_edges_propagation_tree=1000,
                 max_edges_hidden_network=4000, wordcloud_max_words=15,
                 max_cascades=23):
        self.dataset = dataset
        self.data = data
        cast_strings_to_timestamps(self.data)
        self.host = host
        self.port = port
        self.reference_types = reference_types
        self.prepopulator = Prepopulator(
            host=self.host,
            port=self.port,
            reference_types=self.reference_types,
            graph_layout=graph_layout,
            propagation_threshold=propagation_threshold,
            propagation_frequency=propagation_frequency,
            max_edges_propagation_tree=max_edges_propagation_tree,
            max_edges_hidden_network=max_edges_hidden_network,
            wordcloud_max_words=wordcloud_max_words,
            available_datasets=[self.dataset],
            erase_existing=False,
            max_cascades=max_cascades,
            modules=('egonet', 'layout', 'diffusion', 'diffusion_static_plots', 'network', 'histogram',
                     'wordcloud', 'tweet_table'))

    def process(self):
        logger.info(f"Processing dataset {self.dataset} with data {self.data}")
        # Send data to raw collection
        logger.info(f"Storing data in raw collection")
        try:
            self.store_raw_data()
        except Exception as e:
            logger.error(f"Error storing data in raw collection: {e}")
            raise RuntimeError(f"Error storing data in raw collection: {e}") from e

        # Prepopulate the database
        logger.info(f"Prepopulating database")
        try:
            self.prepopulator.prepopulate()
        except Exception as e:
            logger.error(f"Error prepopulating database: {e}")

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
