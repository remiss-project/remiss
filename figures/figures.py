import logging
from abc import ABC

from pymongo import MongoClient
from pymongoarrow.monkey import patch_all

patch_all()

logger = logging.getLogger(__name__)


class MongoPlotFactory(ABC):
    def __init__(self, host="localhost", port=27017, available_datasets=None):
        super().__init__()
        self.host = host
        self.port = port
        self._available_datasets = available_datasets
        self._min_max_dates = {}


    def _validate_dataset(self, client, dataset):
        if dataset not in client.list_database_names():
            raise RuntimeError(f'Dataset {dataset} not found')
        else:
            collections = client.get_database(dataset).list_collection_names()
            if 'raw' not in collections:
                raise RuntimeError(f'Collection raw not found in dataset {dataset}')

    def get_user_id(self, dataset, username):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        tweet = collection.find_one({'author.username': username})
        client.close()
        if tweet:
            return tweet['author']['id']
        else:
            raise RuntimeError(f'User {username} not found')

    def get_username(self, dataset, user_id):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        tweet = collection.find_one({'author.id': user_id})
        client.close()
        if tweet:
            return tweet['author']['username']
        else:
            raise RuntimeError(f'User {user_id} not found')

    @property
    def available_datasets(self):
        if not self._available_datasets:
            client = MongoClient(self.host, self.port)
            available_datasets = client.list_database_names()
            client.close()
            available_datasets = list(set(available_datasets) - {'admin', 'config', 'local', 'CVCUI2'})
            if not available_datasets:
                raise RuntimeError('No datasets available')
            self._available_datasets = available_datasets
        return self._available_datasets

    def get_date_range(self, dataset):
        if dataset not in self._min_max_dates:
            client = MongoClient(self.host, self.port)
            self._validate_dataset(client, dataset)
            database = client.get_database(dataset)
            dataset = database.get_collection('raw')
            self._min_max_dates[dataset] = self._get_date_range(dataset)
            client.close()
        return self._min_max_dates[dataset]

    @staticmethod
    def _get_date_range(collection):
        min_date_allowed = collection.find_one(sort=[('created_at', 1)])['created_at'].date()
        max_date_allowed = collection.find_one(sort=[('created_at', -1)])['created_at'].date()
        return min_date_allowed, max_date_allowed
