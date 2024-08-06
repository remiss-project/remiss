import datetime
import logging
import time

import igraph as ig
import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.schema import Schema

from propagation.base import BasePropagationMetrics

logger = logging.getLogger(__name__)


class Egonet(BasePropagationMetrics):
    def __init__(self, host='localhost', port=27017, threshold=0, delete_vertices=True,
                 reference_types=('retweeted', 'quoted')):
        super().__init__(host, port, reference_types)
        self.threshold = threshold
        self.delete_vertices = delete_vertices
        self._hidden_networks = {}
        self._hidden_network_backbones = {}

    def get_egonet(self, dataset, author_id, depth, start_date=None, end_date=None, hashtags=None):
        """
        Returns the egonet of a user of a certain date and depth if present,
        otherwise returns the simplified hidden network
        :param dataset:
        :param author_id:
        :param depth:
        :return:
        """
        hidden_network = self.get_hidden_network(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)
        node = hidden_network.vs.find(author_id=author_id)
        neighbours = hidden_network.neighborhood(node, order=depth)
        egonet = hidden_network.induced_subgraph(neighbours)
        return egonet

    def get_hidden_network(self, dataset, start_date=None, end_date=None, hashtags=None):
        start_date, end_date = self._validate_dates(dataset, start_date, end_date)
        if start_date or end_date or hashtags:
            return self._compute_hidden_network(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)
        if dataset not in self._hidden_networks:
            network = self._compute_hidden_network(dataset)
            self._hidden_networks[dataset] = network

        return self._hidden_networks[dataset]

    def get_hidden_network_backbone(self, dataset, start_date=None, end_date=None, hashtags=None):
        start_date, end_date = self._validate_dates(dataset, start_date, end_date)
        if start_date or end_date or hashtags:
            return self._compute_hidden_network_backbone(dataset, start_date=start_date, end_date=end_date,
                                                         hashtags=hashtags)
        if dataset not in self._hidden_network_backbones:
            network = self._compute_hidden_network_backbone(dataset)
            self._hidden_network_backbones[dataset] = network

        return self._hidden_network_backbones[dataset]

    def _validate_dates(self, dataset, start_date, end_date):
        dataset_start_date, dataset_end_date = self.get_datatset_data_range(dataset)
        if start_date:
            start_date = pd.to_datetime(start_date).date()
        if end_date:
            end_date = pd.to_datetime(end_date).date()

        if start_date == dataset_start_date:
            start_date = None
        if end_date == dataset_end_date:
            end_date = None

        return start_date, end_date

    def get_datatset_data_range(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        min_date_allowed = collection.find_one(sort=[('created_at', 1)])['created_at'].date()
        max_date_allowed = collection.find_one(sort=[('created_at', -1)])['created_at'].date()
        client.close()
        return min_date_allowed, max_date_allowed

    def _compute_hidden_network_backbone(self, dataset, start_date=None, end_date=None, hashtags=None):
        hidden_network = self.get_hidden_network(dataset, start_date=start_date, end_date=end_date, hashtags=hashtags)
        backbone = self._simplify_graph(hidden_network)

        return backbone

    def _get_references(self, dataset, start_date=None, end_date=None, hashtags=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        references_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'source': '$author.id', 'target': '$referenced_tweets.author.id'}},
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
        self._add_filters(references_pipeline, start_date, end_date, hashtags)
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
        authors = pd.DataFrame({'author_id': np.unique(references[['source', 'target']].values)})

        if len(references) == 0:
            # in case of no authors we return an empty graph
            return ig.Graph(directed=True)

        logger.debug('Computing graph')
        start_time = time.time()
        # switch id by position (which will be the node id in the graph) and set it as index
        author_to_id = authors['author_id'].reset_index().set_index('author_id')
        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['author_id'] = authors['author_id']

        if len(references) > 0:
            # convert references which are author id based to graph id based
            references['source'] = author_to_id.loc[references['source']].reset_index(drop=True)
            references['target'] = author_to_id.loc[references['target']].reset_index(drop=True)

            g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
            g.es['weight'] = references['weight']
            g.es['weight_inv'] = references['weight_inv']
            g.es['weight_norm'] = references['weight_norm']

        logger.debug(g.summary())
        logger.debug(f'Graph computed in {time.time() - start_time} seconds')

        return g

    def _simplify_graph(self, network):
        if network.vcount() == 0:
            return network
        network = self.compute_backbone(network, self.threshold, self.delete_vertices)

        return network

    def persist(self, datasets):
        # Save to mongodb
        for dataset in datasets:
            hidden_network = self.get_hidden_network(dataset)
            hidden_network_backbone = self.get_hidden_network_backbone(dataset)
            self._persist_graph_to_mongodb(hidden_network, dataset, 'hidden_network')
            self._persist_graph_to_mongodb(hidden_network_backbone, dataset, 'hidden_network_backbone')

    def load_from_mongodb(self, datasets):
        for dataset in datasets:
            try:
                hidden_network = self._load_graph_from_mongodb(dataset, 'hidden_network')
                hidden_network_backbone = self._load_graph_from_mongodb(dataset, 'hidden_network_backbone')
                self._hidden_networks[dataset] = hidden_network
                self._hidden_network_backbones[dataset] = hidden_network_backbone
            except Exception as ex:
                logger.error(f'Error loading hidden network for dataset {dataset} with error {ex}')

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
        collection = database.get_collection(f'{collection_name}_edges')
        references = collection.aggregate_pandas_all([])
        collection = database.get_collection(f'{collection_name}_vertices')
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
    def compute_alphas(graph):
        # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
        weights = np.array(graph.es['weight_norm'])
        degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
        alphas = (1 - weights) ** (degrees - 1)
        return alphas

    @staticmethod
    def compute_backbone(graph, alpha=0.05, delete_vertices=True):
        alphas = Egonet.compute_alphas(graph)
        good = np.nonzero(alphas > alpha)[0]
        backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)


        return backbone

    @staticmethod
    def _add_filters(pipeline, start_date, end_date, hashtags):
        if start_date:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})
        if end_date:
            pipeline.insert(0, {'$match': {'created_at': {'$lt': pd.to_datetime(end_date)}}})
        if hashtags:
            # filter if it has at least a hashtag in the list
            pipeline.insert(0, {'$match': {'entities.hashtags.tag': {'$in': hashtags}}})
