import logging
import time

import igraph as ig
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.schema import Schema

from propagation.base import BasePropagationMetrics

logger = logging.getLogger(__name__)


class Egonet(BasePropagationMetrics):
    def __init__(self, host='localhost', port=27017, delete_vertices=True,
                 reference_types=('retweeted', 'quoted', 'reply_to'), include_unknown_authors=False):
        super().__init__(host, port, reference_types)
        self.include_unknown_authors = include_unknown_authors
        self.delete_vertices = delete_vertices
        self._hidden_network_cache = {}
        self.hidden_network_backbone_cache = {}

    def get_egonet(self, dataset, author_id, depth, start_date=None, end_date=None, hashtags=None):
        """
        Returns the egonet of a user of a certain date and depth if present,
        otherwise returns the simplified hidden network
        :param dataset:
        :param author_id:
        :param depth:
        :return:
        """
        start_time = time.time()
        logger.debug(f'Getting egonet for {author_id} with depth {depth}')
        hidden_network = self.get_hidden_network(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)
        node = hidden_network.vs.find(author_id=author_id)
        neighbours = hidden_network.neighborhood(node, order=depth)
        egonet = hidden_network.induced_subgraph(neighbours)
        logger.debug(f'Egonet computed in {time.time() - start_time} seconds')
        return egonet

    def get_hidden_network(self, dataset, start_date=None, end_date=None, hashtags=None):
        start_date, end_date = self._validate_dates(dataset, start_date, end_date)
        if start_date or end_date or hashtags:
            network = self._compute_hidden_network(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)
        else:
            if dataset not in self._hidden_network_cache:
                try:
                    self._hidden_network_cache[dataset] = self.load_hidden_network(dataset)
                except ValueError as ex:
                    if 'not found in database' in str(ex):
                        network = self._compute_hidden_network(dataset)
                        self._hidden_network_cache[dataset] = network
                    else:
                        raise ex
            network = self._hidden_network_cache[dataset]

        return network

    def load_hidden_network(self, dataset):
        return self._load_graph_from_mongodb(dataset, 'hidden_network')

    def _validate_dates(self, dataset, start_date, end_date):
        dataset_start_date, dataset_end_date = self.get_dataset_data_range(dataset)
        if start_date:
            start_date = pd.to_datetime(start_date).date()
        if end_date:
            end_date = pd.to_datetime(end_date).date()

        if start_date == dataset_start_date:
            start_date = None
        if end_date == dataset_end_date:
            end_date = None

        return start_date, end_date

    def get_dataset_data_range(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        min_date_allowed = collection.find_one(sort=[('created_at', 1)])['created_at'].date()
        max_date_allowed = collection.find_one(sort=[('created_at', -1)])['created_at'].date()
        client.close()
        return min_date_allowed, max_date_allowed

    def _get_authors(self, dataset, start_date=None, end_date=None, hashtags=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        authors_pipeline = [
            {'$project': {'_id': 0, 'author_id': '$author.id', 'username': '$author.username'}},
            {'$group': {'_id': '$author_id', 'username': {'$first': '$username'}}},
            {'$sort': {'username': 1}},
            {'$project': {'_id': 0, 'author_id': '$_id', 'username': 1}}
        ]
        self._add_filters(authors_pipeline, start_date, end_date, hashtags)
        logger.debug('Computing authors')
        start_time = time.time()
        schema = Schema({'author_id': str, 'username': str})
        authors = collection.aggregate_pandas_all(authors_pipeline, schema=schema)
        logger.debug(f'Authors computed in {time.time() - start_time} seconds')
        client.close()
        return authors

    def _get_references(self, dataset, start_date=None, end_date=None, hashtags=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        references_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True},
                        '$expr': {'$ne': ['$author.id', '$referenced_tweets.author.id']}}},
            {'$project': {'_id': 0, 'target': '$author.id', 'source': '$referenced_tweets.author.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'},
                        'weight': {'$count': {}}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target', 'weight': 1}},
            {'$group': {'_id': '$source',
                        'node_weight': {'$sum': '$weight'},
                        'references': {'$push': {'target': '$target', 'weight': '$weight'}}}},
            {'$unwind': '$references'},
            {'$project': {'_id': 0, 'source': '$_id', 'target': '$references.target',
                          'weight': '$references.weight',
                          'weight_inv': {'$divide': [1, '$references.weight']},
                          'weight_norm': {'$divide': ['$references.weight', '$node_weight']},
                          }},
        ]
        # self._add_filters(references_pipeline, start_date, end_date, hashtags)
        logger.debug('Computing references')
        start_time = time.time()
        schema = Schema({'source': str, 'target': str, 'weight': int, 'weight_inv': float, 'weight_norm': float})
        references = collection.aggregate_pandas_all(references_pipeline, schema=schema)
        logger.debug(f'References computed in {time.time() - start_time} seconds')
        client.close()
        return references

    def _compute_hidden_network(self, dataset, start_date=None, end_date=None, hashtags=None):
        """
        Computes the hidden graph, this is, the graph of users that have interacted with each other.
        :param dataset: collection name within the database where the tweets are stored
        :return: a networkx graph with the users as nodes and the edges representing interactions between users
        """
        references = self._get_references(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)
        authors = self._get_authors(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)

        if len(references) == 0:
            # in case of no authors we return an empty graph
            return ig.Graph(directed=True)

        logger.debug('Computing graph')
        start_time = time.time()
        # switch id by position (which will be the node id in the graph) and set it as index
        author_to_id = authors['author_id'].reset_index().set_index('author_id')
        # drop references if the author is not in the authors list
        references = references[
            references['source'].isin(author_to_id.index) & references['target'].isin(author_to_id.index)]
        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['author_id'] = authors['author_id']
        g.vs['username'] = authors['username']

        if len(references) > 0:
            # convert references which are author id based to graph id based
            references['source'] = author_to_id.loc[references['source']].values
            references['target'] = author_to_id.loc[references['target']].values
            references = references.reset_index(drop=True)
            references = references.dropna(axis=0, subset=['source', 'target'])
            g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
            g.es['weight'] = references['weight']
            g.es['weight_inv'] = references['weight_inv']
            g.es['weight_norm'] = references['weight_norm']

        g.simplify(combine_edges='sum')
        logger.debug(g.summary())
        logger.debug(f'Graph computed in {time.time() - start_time} seconds')

        return g

    def persist(self, datasets):
        # Save to mongodb
        for dataset in datasets:
            logger.info(f'Computing hidden network for {dataset}')
            start_time = time.time()
            hidden_network = self._compute_hidden_network(dataset)
            logger.info(f'Hidden network computed in {time.time() - start_time} seconds')

            self._persist_graph_to_mongodb(hidden_network, dataset, 'hidden_network')

    def _persist_graph_to_mongodb(self, graph, dataset, collection_name):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection(f'{collection_name}_edges')
        collection.drop()
        collection.insert_many([{'source': e.source,
                                 'target': e.target,
                                 'weight': int(e['weight']),
                                 'weight_inv': float(e['weight_inv']),
                                 'weight_norm': float(e['weight_norm'])}
                                for e in graph.es])
        collection = database.get_collection(f'{collection_name}_vertices')
        collection.drop()
        collection.insert_many([{'author_id': v['author_id']} for v in graph.vs])
        client.close()

    def _load_graph_from_mongodb(self, dataset, collection_name):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        edges_collection_name = f'{collection_name}_edges'
        vertices_collection_name = f'{collection_name}_vertices'
        if edges_collection_name not in database.list_collection_names() or vertices_collection_name not in database.list_collection_names():
            logger.error(f'Collection {edges_collection_name} or {vertices_collection_name} not found in database {dataset}')
            raise ValueError(f'Collection {edges_collection_name} or {vertices_collection_name} not found in database {dataset}')
        collection = database.get_collection(edges_collection_name)
        references = collection.aggregate_pandas_all([])
        collection = database.get_collection(vertices_collection_name)
        authors = collection.aggregate_pandas_all([])
        client.close()
        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['author_id'] = authors['author_id']
        g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
        g.es['weight'] = references['weight']
        g.es['weight_inv'] = references['weight_inv']
        g.es['weight_norm'] = references['weight_norm']

        return g

    @staticmethod
    def _add_filters(pipeline, start_date, end_date, hashtags):
        if start_date:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})
        if end_date:
            pipeline.insert(0, {'$match': {'created_at': {'$lt': pd.to_datetime(end_date)}}})
        if hashtags:
            # filter if it has at least a hashtag in the list
            pipeline.insert(0, {'$match': {'entities.hashtags.tag': {'$in': hashtags}}})
