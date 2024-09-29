import logging

import pandas as pd
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all

patch_all()

from figures.figures import MongoPlotFactory

logger = logging.getLogger(__name__)


class ControlPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None, max_wordcloud_words=150):
        super().__init__(host, port, available_datasets)
        self._available_hashtags = {}
        self.max_wordcloud_words = max_wordcloud_words

    def get_hashtag_freqs(self, dataset, author_id=None, start_date=None, end_date=None):
        if author_id is None and start_date is None and end_date is None:
            if dataset not in self._available_hashtags:
                self._available_hashtags[dataset] = self._get_dataset_hashtag_freqs(dataset)
            return self._available_hashtags[dataset]
        else:
            return self._compute_hashtag_freqs(dataset, author_id, start_date, end_date)

    def _get_dataset_hashtag_freqs(self, dataset):
        try:
            hashtag_freqs = self._load_hashtag_freqs(dataset)
        except Exception as e:
            logger.error(f"Error loading hashtag frequencies for dataset {dataset}: {e}")
            hashtag_freqs = self._compute_hashtag_freqs(dataset)
        return hashtag_freqs

    def _compute_hashtag_freqs(self, dataset, author_id=None, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        pipeline = [
            {'$unwind': '$entities.hashtags'},
            {'$group': {'_id': '$entities.hashtags.tag', 'count': {'$count': {}}}},
            {'$sort': {'count': -1}},
            {'$project': {'_id': 0, 'hashtag': '$_id', 'count': 1}}
        ]
        if author_id:
            pipeline.insert(1, {'$match': {'author.id': author_id}})
        if start_date:
            pipeline.insert(1, {'$match': {'created_at': {'$gte': pd.Timestamp(start_date)}}})
        if end_date:
            pipeline.insert(1, {'$match': {'created_at': {'$lte': pd.Timestamp(end_date)}}})
        available_hashtags_freqs = collection.aggregate_pandas_all(pipeline)
        if self.max_wordcloud_words:
            max_wordcloud_words = min(self.max_wordcloud_words, len(available_hashtags_freqs))
            available_hashtags_freqs = available_hashtags_freqs[:max_wordcloud_words]
        client.close()
        return available_hashtags_freqs

    def persist(self, datasets):
        for dataset in datasets:
            logger.info(f'Generating {self.__class__.__name__} for dataset {dataset}')
            self._persist_hashtag_freqs(dataset)

    def _persist_hashtag_freqs(self, dataset):
        hashtag_freqs = self._compute_hashtag_freqs(dataset)
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('hashtag_freqs')
        collection.drop()
        collection.insert_many(hashtag_freqs.to_dict(orient='records'))
        client.close()

    def _load_hashtag_freqs(self, dataset):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('hashtag_freqs')
        pipeline = [
            {'$sort': {'count': -1}},
            {'$project': {'_id': 0, 'hashtag': 1, 'count': 1}}
        ]
        hashtag_freqs = collection.aggregate_pandas_all(pipeline)
        client.close()
        return hashtag_freqs