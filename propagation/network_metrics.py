import logging
import time

import pandas as pd
from pymongo import MongoClient

logger = logging.getLogger(__name__)


class NetworkMetrics:
    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted'), frequency='1D'):
        self.reference_types = reference_types
        self.host = host
        self.port = port
        self.bin_size = int(frequency[:-1])
        pd_units = {'D': 'day', 'W': 'week', 'M': 'month', 'Y': 'year'}
        self.unit = pd_units[frequency[-1]]

    def get_legitimacy(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': '$author.id',
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'legitimacy': 1}},
        ]
        logging.info('Computing legitimacy')
        start_time = time.time()
        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        legitimacy = legitimacy.set_index('author_id')
        legitimacy = legitimacy.sort_values('legitimacy', ascending=False)
        logging.info(f'Legitimacy computed in {time.time() - start_time} seconds')
        return legitimacy

    def _get_legitimacy_over_time(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': {'author': '$author.id',
                                'date': {
                                    "$dateTrunc": {'date': "$created_at", 'unit': self.unit, 'binSize': self.bin_size}}
                                },
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id.author',
                          'date': '$_id.date',
                          'legitimacy': 1}},
        ]
        logging.info('Computing reputation')

        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        if len(legitimacy) == 0:
            raise ValueError(
                f'No data available for the selected time range and dataset: {dataset} {self.unit} {self.bin_size}')
        legitimacy = legitimacy.pivot(columns='date', index='author_id', values='legitimacy')
        legitimacy = legitimacy.fillna(0)
        return legitimacy

    def get_reputation(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_over_time(dataset)
        reputation = legitimacy.cumsum(axis=1)

        logging.info(f'Reputation computed in {time.time() - start_time} seconds')
        return reputation

    def get_status(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_over_time(dataset)
        reputation = legitimacy.cumsum(axis=1)
        status = reputation.apply(lambda x: x.argsort())
        logging.info(f'Status computed in {time.time() - start_time} seconds')
        return status
