import random
import time
import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import igraph as ig
import numpy as np
import pandas as pd
import plotly.express as px
from pymongo import MongoClient

from propagation import Egonet
from tests.conftest import create_test_data_from_edges, create_test_data


class TestEgonetCase(unittest.TestCase):

    def setUp(self):
        self.egonet = Egonet()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    @patch('propagation.egonet.MongoClient')
    def test_compute_hidden_network(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline, schema=None):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': ['1', '2', '3'],
                                      'target': ['2', '3', '1'],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges

            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        actual = self.egonet.get_hidden_network(self.test_dataset)

        self.assertEqual({'1', '2', '3'}, set(actual.vs['author_id']))
        edges = {frozenset((actual.vs[s]['author_id'], actual.vs[t]['author_id'])) for s, t in actual.get_edgelist()}
        expected = {('2', '3'), ('1', '2'), ('1', '3')}
        expected = {frozenset(x) for x in expected}
        self.assertEqual(expected, edges)

    @patch('propagation.egonet.MongoClient')
    def test_compute_hidden_network_2(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline, schema=None):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': ['1', '2', '3'],
                                      'target': ['2', '3', '4'],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3', '4']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        actual = self.egonet.get_hidden_network(None)

        self.assertEqual({'1', '2', '3', '4'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('2', '3'), ('1', '2'), ('3', '4')}, edges)

    @patch('propagation.egonet.MongoClient')
    def test_compute_hidden_network_weight(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline, schema=None):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': ['1', '2', '3'],
                                      'target': ['2', '3', '4'],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3', '4']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        graph = self.egonet.get_hidden_network(None)

        actual = graph.es['weight']
        expected = [1, 2, 3]
        self.assertEqual(expected, actual)

        actual = graph.es['weight_inv']
        expected = [1, 0.5, 0.33]
        self.assertEqual(expected, actual)

        actual = graph.es['weight_norm']
        expected = [1, 0.5, 0.5]
        self.assertEqual(expected, actual)

    @patch('propagation.egonet.MongoClient')
    def test_get_egonet(self, mock_mongo_client):
        # Checks it returns the whole thing if the user is not present

        mock_collection = Mock()

        def aggregate_pandas_all(pipeline, schema=None):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': ['1', '2', '3'],
                                      'target': ['2', '3', '4'],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3', '4']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'
        user_id = '1'
        depth = 1

        actual = self.egonet.get_egonet(collection, user_id, depth)

        self.assertEqual({'1', '2'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2')}, edges)

    def test__get_references(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4']})
        test_data = create_test_data_from_edges(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet._get_references(self.tmp_dataset)
        actual = actual.sort_values(['source', 'target']).reset_index(drop=True)
        self.assertEqual(actual['weight'].to_list(), [1, 1, 2, 1])
        self.assertEqual(actual['weight_inv'].to_list(), [1, 1, 0.5, 1])
        self.assertEqual(actual['weight_norm'].to_list(), [0.5, 0.5, 1, 1])
        self.assertEqual(actual['source'].to_list(), ['1', '1', '2', '3'])
        self.assertEqual(actual['target'].to_list(), ['2', '4', '3', '4'])

    def test_compute_hidden_network_speed(self):
        # Checks it returns the whole thing if the user is not present
        test_data = create_test_data(1000)
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')

        collection.insert_many(test_data)

        print('computing egonet')
        self.egonet.host = 'localhost'
        self.egonet.port = 27017
        # time computation of get_egonet
        start_time = time.time()
        actual = self.egonet._compute_hidden_network(self.tmp_dataset)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLessEqual(total_time, 10)

    def test_backbone(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '2', '1', '1', '1', '1', '1'],
                                       'target': ['2', '3', '3', '2', '2', '2', '2', '4']})
        test_data = create_test_data_from_edges(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        network = self.egonet._compute_hidden_network(self.tmp_dataset)
        backbone = self.egonet.compute_backbone(network, alpha=0.4)
        actual = {frozenset(x) for x in backbone.get_edgelist()}
        expected = {frozenset((0, 1))}
        self.assertEqual(expected, actual)

    def test_backbone_2(self):
        # Create a test graph
        alpha = 0.05

        network = ig.Graph.Erdos_Renyi(250, 0.02, directed=False)
        network.es["weight_norm"] = np.random.uniform(0, 0.5, network.ecount())

        # Test the backbone filter
        backbone = self.egonet.compute_backbone(network, alpha=alpha)

        # Ensure that all edge weights are below alpha
        for edge in backbone.get_edgelist():
            weight = backbone.es[backbone.get_eid(*edge)]['weight_norm']
            degree = backbone.degree(edge[0])
            edge_alpha = (1 - weight) ** (degree - 1)
            self.assertGreater(edge_alpha, alpha)

    @patch('propagation.egonet.MongoClient')
    def test_compute_hidden_network_backbone(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline, schema=None):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': ['1', '2', '3'],
                                      'target': ['2', '3', '1'],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges

            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        actual = self.egonet.get_hidden_network_backbone(self.test_dataset)

        self.assertEqual({'1', '2', '3'}, set(actual.vs['author_id']))
        edges = {frozenset((actual.vs[s]['author_id'], actual.vs[t]['author_id'])) for s, t in actual.get_edgelist()}
        expected = {('2', '3'), ('1', '3')}
        expected = {frozenset(x) for x in expected}
        self.assertEqual(expected, edges)

    def test_persistence_and_loading(self):
        # Test the persistence and loading of the graph
        self.egonet.host = 'localhost'
        self.egonet.port = 27017
        expected_edges = pd.DataFrame({'source': ['1', '2', '2', '1', '1', '1', '1', '1'],
                                       'target': ['2', '3', '3', '2', '2', '2', '2', '4']})
        test_data = create_test_data_from_edges(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        self.egonet.persist([self.tmp_dataset])

        start_time = time.time()
        self.egonet.load_from_mongodb([self.tmp_dataset])
        end_time = time.time()
        print(f'loaded in {end_time - start_time} seconds')
        start_time = time.time()

        expected_hidden_network = self.egonet._compute_hidden_network(self.tmp_dataset)
        expected_hidden_network_backbone = self.egonet.compute_backbone(expected_hidden_network,
                                                                        alpha=self.egonet.threshold)

        end_time = time.time()
        print(f'computed in {end_time - start_time} seconds')

        actual_hidden_network = self.egonet._hidden_networks[self.tmp_dataset]
        actual_hidden_network_backbone = self.egonet._hidden_network_backbones[self.tmp_dataset]

        self.assertEqual(expected_hidden_network.vcount(), actual_hidden_network.vcount())
        self.assertEqual(expected_hidden_network.ecount(), actual_hidden_network.ecount())
        self.assertEqual(expected_hidden_network_backbone.vcount(), actual_hidden_network_backbone.vcount())
        self.assertEqual(expected_hidden_network_backbone.ecount(), actual_hidden_network_backbone.ecount())

    # Test dataset

    def test_compute_hidden_network_full(self):
        actual = self.egonet.get_hidden_network(self.test_dataset)

        self.assertEqual(actual.vcount(), 3315)
        self.assertEqual(actual.ecount(), 5844)

    def test_compute_hidden_network_weight_full(self):
        graph = self.egonet.get_hidden_network(self.test_dataset)

        actual = pd.Series(graph.es['weight']).value_counts()[:5].to_list()
        expected = [4932, 430, 136, 99, 49]
        self.assertEqual(expected, actual)

        actual = pd.Series(graph.es['weight_inv']).value_counts().to_list()[:5]
        expected = [4932, 430, 136, 99, 49]
        self.assertEqual(expected, actual)

        actual = pd.Series(graph.es['weight_norm']).value_counts().to_list()[:5]
        expected = [2401, 157, 139, 121, 88]
        self.assertEqual(expected, actual)

    def test_get_egonet_full(self):
        user_id = '999321854'
        depth = 1

        actual = self.egonet.get_egonet(self.test_dataset, user_id, depth)

        actual_nodes = set(actual.vs['author_id'])
        expected_nodes = {'270839361', '999321854'}

        self.assertEqual(expected_nodes, actual_nodes)

        actual_edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        expected_edges = {('999321854', '270839361')}

        self.assertEqual(expected_edges, actual_edges)

    def test_get_egonet_missing_user_full(self):
        user_id = '1'
        depth = 2

        actual = self.egonet.get_egonet(self.test_dataset, user_id, depth)

        # check it returns the full hidden network
        self.assertEqual(actual.vcount(), 3315)
        self.assertEqual(actual.ecount(), 5844)

    def test_get_egonet_missing_user_backbone(self):
        user_id = '1'
        depth = 2
        self.egonet.threshold = 0.4

        actual = self.egonet.get_egonet(self.test_dataset, user_id, depth)

        # check it returns the hidden network backbone
        self.assertEqual(actual.vcount(), 3224)
        self.assertEqual(actual.ecount(), 4801)

    def test__get_references_full(self):
        actual = self.egonet._get_references(self.test_dataset)
        actual = actual.sort_values(['source', 'target']).reset_index(drop=True)
        expected = [4932, 430, 136, 99, 49, 30, 28, 19, 15, 15, 14, 12, 10, 9, 7, 7, 7, 6, 5, 4, 3, 3, 2, 1, 1]
        expected_weight_norm = [2401, 157, 139, 121, 88, 87, 78, 77, 74, 72]
        self.assertEqual(actual['weight'].value_counts().to_list(), expected)
        self.assertEqual(actual['weight_inv'].value_counts().to_list(), expected)
        self.assertEqual(actual['weight_norm'].value_counts().to_list()[:10], expected_weight_norm)
        self.assertEqual(actual['source'].to_list()[:2], ['1000010057743065089', '1000300778190528512'])
        self.assertEqual(actual['target'].to_list()[:2], ['270839361', '420374996'])

    def test_compute_hidden_network_speed_full(self):
        print('computing hidden network')
        self.egonet.host = 'localhost'
        self.egonet.port = 27017
        # time computation of get_egonet
        start_time = time.time()
        actual = self.egonet._compute_hidden_network(self.test_dataset)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLessEqual(total_time, 10)

    def test_backbone_full(self):
        network = self.egonet._compute_hidden_network(self.test_dataset)
        backbone = self.egonet.compute_backbone(network, alpha=0.95)
        self.assertEqual(backbone.vcount(), 2522)
        self.assertEqual(backbone.ecount(), 2366)

    def test_backbone_full_nothing(self):
        network = self.egonet._compute_hidden_network(self.test_dataset)
        backbone = self.egonet.compute_backbone(network, alpha=1)
        self.assertEqual(backbone.vcount(), 0)
        self.assertEqual(backbone.ecount(), 0)

    def test_show_alpha_distribution(self):
        network = self.egonet._compute_hidden_network(self.test_dataset)

        alphas = self.egonet.compute_alphas(network)
        # plot alphas histogram with plotly
        fig = px.histogram(alphas, nbins=1000)
        fig.update_xaxes(title_text='Alpha')
        fig.update_yaxes(title_text='Count')
        fig.show()

    def test_persistence_and_loading_full(self):
        # Test the persistence and loading of the graph
        self.egonet.persist([self.test_dataset])

        start_time = time.time()
        self.egonet.load_from_mongodb([self.test_dataset])
        end_time = time.time()
        print(f'loaded in {end_time - start_time} seconds')
        start_time = time.time()

        expected_hidden_network = self.egonet._compute_hidden_network(self.test_dataset)
        expected_hidden_network_backbone = self.egonet.compute_backbone(expected_hidden_network,
                                                                        alpha=self.egonet.threshold)

        end_time = time.time()
        print(f'computed in {end_time - start_time} seconds')

        actual_hidden_network = self.egonet._hidden_networks[self.test_dataset]
        actual_hidden_network_backbone = self.egonet._hidden_network_backbones[self.test_dataset]

        self.assertEqual(expected_hidden_network.vcount(), actual_hidden_network.vcount())
        self.assertEqual(expected_hidden_network.ecount(), actual_hidden_network.ecount())
        self.assertEqual(expected_hidden_network_backbone.vcount(), actual_hidden_network_backbone.vcount())
        self.assertEqual(expected_hidden_network_backbone.ecount(), actual_hidden_network_backbone.ecount())





if __name__ == '__main__':
    unittest.main()
