import logging
import time

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongoarrow.schema import Schema

logger = logging.getLogger(__name__)
import igraph as ig


class Egonet:
    def __init__(self, host='localhost', port=27017, simplification=None, threshold=0.05,
                 delete_vertices=True, reference_types=('retweeted', 'quoted')):
        self.reference_types = reference_types
        self.host = host
        self.port = port
        self.simplification = simplification
        self.threshold = threshold
        self.delete_vertices = delete_vertices
        self._hidden_networks = {}
        self._hidden_network_backbones = {}
        self._layouts = {}

    def get_egonet(self, dataset, author_id, depth):
        """
        Returns the egonet of a user of a certain date and depth if present,
        otherwise returns the simplified hidden network
        :param dataset:
        :param author_id:
        :param depth:
        :return:
        """
        hidden_network = self.get_hidden_network(dataset)
        # check if the user is in the hidden network
        if author_id:
            try:
                node = hidden_network.vs.find(author_id=author_id)
                neighbours = hidden_network.neighborhood(node, order=depth)
                egonet = hidden_network.induced_subgraph(neighbours)
                return egonet
            except (RuntimeError, ValueError) as ex:
                logger.debug(f'Computing neighbourhood for user {author_id} failed with error {ex}')
        if self.simplification:
            return self.get_hidden_network_backbone(dataset)
        else:
            return hidden_network

    def get_hidden_network(self, dataset):
        if dataset not in self._hidden_networks:
            network = self._compute_hidden_network(dataset)
            self._hidden_networks[dataset] = network

        return self._hidden_networks[dataset]

    def get_hidden_network_backbone(self, dataset):
        if dataset not in self._hidden_network_backbones:
            network = self._compute_hidden_network_backbone(dataset)
            self._hidden_network_backbones[dataset] = network

        return self._hidden_network_backbones[dataset]

    def _compute_hidden_network_backbone(self, dataset):
        hidden_network = self.get_hidden_network(dataset)
        backbone = self._simplify_graph(hidden_network)

        return backbone

    def _get_authors(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        nested_pipeline = [
            {'$project': {'_id': 0, 'author_id': '$author.id'}
             }]
        self._add_date_filters(nested_pipeline, start_date, end_date)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'author_id': '$referenced_tweets.author.id'}},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing authors
            {'$group': {'_id': '$author_id'}},
            {'$project': {'_id': 0, 'author_id': '$_id'}}
        ]
        self._add_date_filters(node_pipeline, start_date, end_date)
        logger.info('Computing authors')
        start_time = time.time()
        schema = Schema({'author_id': str})
        authors = collection.aggregate_pandas_all(node_pipeline, schema=schema)
        logger.info(f'Authors computed in {time.time() - start_time} seconds')
        client.close()
        return authors

    def _get_references(self, dataset, start_date=None, end_date=None):
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
        self._add_date_filters(references_pipeline, start_date, end_date)
        logger.info('Computing references')
        start_time = time.time()
        schema = Schema({'source': str, 'target': str, 'weight': int, 'weight_inv': float, 'weight_norm': float})
        references = collection.aggregate_pandas_all(references_pipeline, schema=schema)
        logger.info(f'References computed in {time.time() - start_time} seconds')
        client.close()
        return references

    def _compute_hidden_network(self, dataset):
        """
        Computes the hidden graph, this is, the graph of users that have interacted with each other.
        :param dataset: collection name within the database where the tweets are stored
        :return: a networkx graph with the users as nodes and the edges representing interactions between users
        """
        authors = self._get_authors(dataset)
        references = self._get_references(dataset)

        if len(authors) == 0:
            # in case of no authors we return an empty graph
            return ig.Graph(directed=True)

        logger.info('Computing graph')
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

        logger.info(g.summary())
        logger.info(f'Graph computed in {time.time() - start_time} seconds')

        return g

    def _simplify_graph(self, network):
        network = self.compute_backbone(network, self.threshold, self.delete_vertices)

        return network

    @staticmethod
    def compute_backbone(graph, alpha=0.05, delete_vertices=True):
        # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
        weights = np.array(graph.es['weight_norm'])
        degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
        alphas = (1 - weights) ** (degrees - 1)
        good = np.nonzero(alphas > alpha)[0]
        backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)

        return backbone

    @staticmethod
    def _add_date_filters(pipeline, start_date, end_date):
        if start_date:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})
        if end_date:
            pipeline.insert(0, {'$match': {'created_at': {'$lt': pd.to_datetime(end_date)}}})
