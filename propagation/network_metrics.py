import logging
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from scipy.stats import rankdata

from propagation.base import BasePropagationMetrics

patch_all()
logger = logging.getLogger(__name__)


class NetworkMetrics(BasePropagationMetrics):
    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted'), frequency='1D',
                 cut_bins=('Low', 'Medium', 'High')):
        super().__init__(host, port, reference_types)
        self.cut_bins = cut_bins
        self.bin_size = int(frequency[:-1])
        pd_units = {'D': 'day', 'W': 'week', 'M': 'month', 'Y': 'year'}
        self.unit = pd_units[frequency[-1]]

    def get_legitimacy(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        legitimacy = collection.aggregate_pandas_all([{'$project': {'_id': 0, 'author_id': 1, 'legitimacy': 1}}])
        legitimacy = legitimacy.set_index('author_id')['legitimacy']
        return legitimacy

    def get_reputation(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        reputation = collection.aggregate_pandas_all([{'$project': {'_id': 0, 'author_id': 1, 'reputation': 1}}])
        reputation = pd.DataFrame(list(reputation['reputation']), index=reputation['author_id'])
        reputation.columns = pd.to_datetime(reputation.columns)
        reputation.columns.name = 'date'
        return reputation

    def get_status(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        status = collection.aggregate_pandas_all([{'$project': {'_id': 0, 'author_id': 1, 'status': 1}}])
        status = pd.DataFrame(list(status['status']), index=status['author_id'])
        status.columns = pd.to_datetime(status.columns)
        status.columns.name = 'date'
        return status

    def compute_legitimacy(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        positive_legitimacy_pipeline = [
            {'$unwind': {'path': '$referenced_tweets'}},
            {'$group': {'_id': '$referenced_tweets.author.id',
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'legitimacy': 1
                          }},
        ]
        logger.debug('Computing legitimacy')
        start_time = time.time()
        legitimacy = collection.aggregate_pandas_all(positive_legitimacy_pipeline)
        legitimacy = legitimacy.set_index('author_id')
        legitimacy = legitimacy['legitimacy']

        # All pipeline
        all_author_ids_pipeline  = [
            {'$group': {'_id': '$author.id'}},
            {'$project': {'_id': 0, 'author_id': '$_id'}},
        ]
        all_author_ids = collection.aggregate_pandas_all(all_author_ids_pipeline)
        legitimacy = legitimacy.reindex(all_author_ids['author_id'])
        legitimacy = legitimacy.fillna(0)

        # Drop na indexes
        legitimacy = legitimacy[~legitimacy.index.isna()]
        legitimacy = legitimacy.sort_values(ascending=False)
        logger.debug(f'Legitimacy computed in {time.time() - start_time} seconds')
        return legitimacy

    def _get_legitimacy_over_time(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        node_pipeline = [
            {'$unwind': {'path': '$referenced_tweets'}},
            {'$group': {'_id': {'author': '$referenced_tweets.author.id',
                                'date': {
                                    "$dateTrunc": {'date': "$created_at", 'unit': self.unit, 'binSize': self.bin_size}}
                                },
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id.author',
                          'date': '$_id.date',
                          'legitimacy': 1}},
        ]
        logger.debug('Computing reputation')

        legitimacy = collection.aggregate_pandas_all(node_pipeline)

        # All pipeline
        all_author_ids_pipeline = [
            {'$group': {'_id': '$author.id'}},
            {'$project': {'_id': 0, 'author_id': '$_id'}},
        ]
        all_author_ids = collection.aggregate_pandas_all(all_author_ids_pipeline)

        if len(legitimacy) == 0:
            raise ValueError(
                f'No data available for the selected time range and dataset: {dataset} {self.unit} {self.bin_size}')
        legitimacy = legitimacy.pivot(columns='date', index='author_id', values='legitimacy')
        # Drop na indexes
        legitimacy = legitimacy[~legitimacy.index.isna()]
        legitimacy = legitimacy.reindex(all_author_ids['author_id'])
        return legitimacy

    def compute_reputation(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_over_time(dataset)
        legitimacy = legitimacy.fillna(0)
        reputation = legitimacy.cumsum(axis=1)
        reputation.name = 'reputation'

        logger.debug(f'Reputation computed in {time.time() - start_time} seconds')
        return reputation

    def compute_status(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_over_time(dataset)
        reputation = legitimacy.fillna(0).cumsum(axis=1)
        status = reputation.apply(lambda x: rankdata(x, method='min', nan_policy='omit'))
        logger.debug(f'Status computed in {time.time() - start_time} seconds')
        return status - 1

    def persist(self, datasets):
        for dataset in datasets:
            logger.info(f'Computing network metrics for {dataset}')
            logger.info('Computing legitimacy')
            legitimacy = self.compute_legitimacy(dataset)
            logger.info('Computing reputation')
            reputation = self.compute_reputation(dataset)
            logger.info('Computing status')
            status = self.compute_status(dataset)
            logger.info('Persisting metrics')
            self._persist_metrics(dataset, legitimacy, reputation, status)

    def get_level(self, data):
        data = data.copy()
        data[data == 0] = np.nan
        try:
            levels = pd.qcut(data, len(self.cut_bins), labels=self.cut_bins)
        except ValueError:
            levels = pd.qcut(data, len(self.cut_bins), duplicates='drop')
            if len(levels.cat.categories) == 2:
                labels = ['Low', 'High']
            elif len(levels.cat.categories) == 1:
                labels = ['Low']
            else:
                labels = self.cut_bins
            levels = levels.cat.rename_categories(labels)
        if data.isna().any():
            # put unknown as the smallest
            categories = levels.cat.categories.to_list()
            levels = levels.cat.add_categories('Null')
            levels = levels.cat.reorder_categories(['Null'] + categories)
            levels = levels.fillna('Null')
        return levels

    def _persist_metrics(self, dataset, legitimacy, reputation, status):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        collection.drop()

        average_reputation = reputation.mean(axis=1)
        average_status = status.mean(axis=1)

        legitimacy_level = self.get_level(legitimacy)
        reputation_level = self.get_level(average_reputation)
        status_level = self.get_level(average_status)

        reputation.columns = reputation.columns.astype(str)
        status.columns = status.columns.astype(str)
        data = []
        for author_id in legitimacy.index:
            current_average_reputation = average_reputation.loc[author_id]
            current_average_status = average_status.loc[author_id]

            current_legitimacy_level = legitimacy_level.loc[author_id]
            current_reputation_level = reputation_level.loc[author_id]
            current_status_level = status_level.loc[author_id]

            current_reputation = reputation.loc[author_id].to_dict()
            current_status = status.loc[author_id].to_dict()

            data.append({'author_id': author_id,
                         'legitimacy': float(legitimacy[author_id]),
                         'reputation': current_reputation,
                         'status': current_status,
                         'average_reputation': current_average_reputation,
                         'average_status': current_average_status,
                         'legitimacy_level': current_legitimacy_level,
                         'reputation_level': current_reputation_level,
                         'status_level': current_status_level})
        collection.insert_many(data)
        client.close()

    def _serialize_date_metric(self, metric, name):
        try:
            metric.index = metric.index.astype(str)
            metric = metric.to_dict()
        except AttributeError:
            logger.error(f'Error serializing {name}')
        return metric

    def load_legitimacy_for_author(self, dataset, author_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        pipeline = [
            {'$match': {'author_id': author_id}},
            {'$project': {'_id': 0, 'legitimacy': 1}}
        ]
        legitimacy = collection.aggregate_pandas_all(pipeline)
        return legitimacy['legitimacy'].values[0]

    def load_reputation_for_author(self, dataset, author_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        pipeline = [
            {'$match': {'author_id': author_id}},
            {'$project': {'_id': 0, 'author_id':1 , 'reputation': 1}}
        ]
        reputation = collection.aggregate_pandas_all(pipeline)
        reputation = pd.Series(reputation['reputation'].values[0], name=reputation['author_id'].values[0])
        return reputation

    def load_status_for_author(self, dataset, author_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        pipeline = [
            {'$match': {'author_id': author_id}},
            {'$project': {'_id': 0, 'author_id': 1, 'status': 1}}
        ]
        status = collection.aggregate_pandas_all(pipeline)
        status = pd.Series(status['status'].values[0], name=status['author_id'].values[0])
        return status