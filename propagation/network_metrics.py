import logging
import time

from pymongo import MongoClient

from propagation.base import BasePropagationMetrics

logger = logging.getLogger(__name__)


class NetworkMetrics(BasePropagationMetrics):
    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted'), frequency='1D'):
        super().__init__(host, port, reference_types)
        self.bin_size = int(frequency[:-1])
        pd_units = {'D': 'day', 'W': 'week', 'M': 'month', 'Y': 'year'}
        self.unit = pd_units[frequency[-1]]

    def compute_legitimacy(self, dataset):
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
        logger.debug('Computing legitimacy')
        start_time = time.time()
        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        legitimacy = legitimacy.set_index('author_id')
        legitimacy = legitimacy.sort_values('legitimacy', ascending=False)
        legitimacy = legitimacy['legitimacy']
        logger.debug(f'Legitimacy computed in {time.time() - start_time} seconds')
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
        logger.debug('Computing reputation')

        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        if len(legitimacy) == 0:
            raise ValueError(
                f'No data available for the selected time range and dataset: {dataset} {self.unit} {self.bin_size}')
        legitimacy = legitimacy.pivot(columns='date', index='author_id', values='legitimacy')
        legitimacy = legitimacy.fillna(0)
        return legitimacy

    def compute_reputation(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_over_time(dataset)
        reputation = legitimacy.cumsum(axis=1)
        reputation.name = 'reputation'

        logger.debug(f'Reputation computed in {time.time() - start_time} seconds')
        return reputation

    def compute_status(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_over_time(dataset)
        reputation = legitimacy.cumsum(axis=1)
        status = reputation.apply(lambda x: x.argsort())
        logger.debug(f'Status computed in {time.time() - start_time} seconds')
        return status

    def persist(self, datasets):
        for dataset in datasets:
            legitimacy = self.compute_legitimacy(dataset)
            reputation = self.compute_reputation(dataset)
            status = self.compute_status(dataset)
            self._persist_metrics(dataset, legitimacy, reputation, status)

    def _persist_metrics(self, dataset, legitimacy, reputation, status):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('network_metrics')
        collection.drop()
        data = []
        for author_id in legitimacy.index:
            average_reputation = float(reputation.loc[author_id].mean())
            average_status = float(status.loc[author_id].mean())
            current_reputation = self._serialize_date_metric(reputation.loc[author_id], 'reputation')
            current_status = self._serialize_date_metric(status.loc[author_id], 'status')

            data.append({'author_id': author_id,
                         'legitimacy': float(legitimacy[author_id]),
                         'reputation': current_reputation,
                         'status': current_status,
                         'average_reputation': average_reputation,
                         'average_status': average_status})
        collection.insert_many(data)
        client.close()

    def _serialize_date_metric(self, metric, name):
        try:
            metric = metric.to_dict()
            metric = {str(k): v for k, v in metric.items()}
        except AttributeError:
            logger.error(f'Error serializing {name}')
        return metric
