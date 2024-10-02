import time
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
from tests.conftest import create_test_data_from_edges, create_test_data, create_test_data_from_edges_with_dates, \
    create_test_data_from_edges_with_hashtags


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
        mock_collection.find_one.return_value = {'created_at': datetime.now()}
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        actual = self.egonet._compute_hidden_network(self.test_dataset)

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

        actual = self.egonet._compute_hidden_network(None)

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

        graph = self.egonet._compute_hidden_network(None)

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
        graph = ig.Graph(directed=True)
        graph.add_vertices(3)
        graph.add_edges([(0, 1), (1, 2)])
        graph.es['weight'] = [1, 2, 3]
        graph.es['weight_inv'] = [1, 0.5, 0.33]
        graph.es['weight_norm'] = [1, 0.5, 0.5]
        graph.vs['author_id'] = ['1', '2', '3']

        self.egonet.get_hidden_network = Mock(return_value=graph)

        user_id = '1'
        depth = 1

        actual = self.egonet.get_egonet(None, user_id, depth)

        self.assertEqual({'1', '2'}, set(actual.vs['author_id']))
        self.assertEqual({('1', '2')},
                         {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()})

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

    def test__get_references_date_filtinering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'created_at': [datetime.now() - timedelta(days=1),
                                                      datetime.now() - timedelta(days=2),
                                                      datetime.now() - timedelta(days=3),
                                                      datetime.now() - timedelta(days=4),
                                                      datetime.now() - timedelta(days=5)]})
        test_data = create_test_data_from_edges_with_dates(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet._get_references(self.tmp_dataset,
                                             start_date=datetime.now() - timedelta(days=3),
                                             end_date=datetime.now() - timedelta(days=1))

        actual = actual.sort_values(['source', 'target']).reset_index(drop=True)
        self.assertEqual(actual['weight'].to_list(), [1, 1])
        self.assertEqual(actual['weight_inv'].to_list(), [1, 1])
        self.assertEqual(actual['weight_norm'].to_list(), [1.0, 1.0])
        self.assertEqual(actual['source'].to_list(), ['1', '2'])
        self.assertEqual(actual['target'].to_list(), ['2', '3'])

    def test__compute_hidden_network_date_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'created_at': [datetime.now() - timedelta(days=1),
                                                      datetime.now() - timedelta(days=2),
                                                      datetime.now() - timedelta(days=3),
                                                      datetime.now() - timedelta(days=4),
                                                      datetime.now() - timedelta(days=5)]})

        test_data = create_test_data_from_edges_with_dates(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet._compute_hidden_network(self.tmp_dataset,
                                                     start_date=datetime.now() - timedelta(days=3),
                                                     end_date=datetime.now() - timedelta(days=1))

        self.assertEqual({'1', '2', '3'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2'), ('2', '3')}, edges)

    def test_get_hidden_network_date_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'created_at': [datetime.now() - timedelta(days=1),
                                                      datetime.now() - timedelta(days=2),
                                                      datetime.now() - timedelta(days=3),
                                                      datetime.now() - timedelta(days=4),
                                                      datetime.now() - timedelta(days=5)]})

        test_data = create_test_data_from_edges_with_dates(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet._compute_hidden_network(self.tmp_dataset,
                                                     start_date=datetime.now() - timedelta(days=3),
                                                     end_date=datetime.now() - timedelta(days=1))

        self.assertEqual({'1', '2', '3'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2'), ('2', '3')}, edges)

    def test_get_egonet_date_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'created_at': [datetime.now() - timedelta(days=1),
                                                      datetime.now() - timedelta(days=2),
                                                      datetime.now() - timedelta(days=3),
                                                      datetime.now() - timedelta(days=4),
                                                      datetime.now() - timedelta(days=5)]})

        test_data = create_test_data_from_edges_with_dates(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        user_id = '1'
        depth = 1

        actual = self.egonet.get_egonet(self.tmp_dataset, user_id, depth,
                                        start_date=datetime.now() - timedelta(days=3),
                                        end_date=datetime.now() - timedelta(days=1))

        self.assertEqual({'1', '2'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2')}, edges)

    def test_get_hidden_network_date_filtering_backbone(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'created_at': [datetime.now() - timedelta(days=1),
                                                      datetime.now() - timedelta(days=2),
                                                      datetime.now() - timedelta(days=3),
                                                      datetime.now() - timedelta(days=4),
                                                      datetime.now() - timedelta(days=5)]})

        test_data = create_test_data_from_edges_with_dates(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet.get_hidden_network_backbone(self.tmp_dataset,
                                                         start_date=datetime.now() - timedelta(days=3),
                                                         end_date=datetime.now() - timedelta(days=1))

        self.assertEqual({'1', '2'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2')}, edges)

    def test__get_references_hashtag_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet._get_references(self.tmp_dataset, hashtags=['#a'])
        actual = actual.sort_values(['source', 'target']).reset_index(drop=True)
        self.assertEqual(actual['weight'].to_list(), [1, 1, 1])
        self.assertEqual(actual['weight_inv'].to_list(), [1, 1, 1])
        self.assertEqual(actual['weight_norm'].to_list(), [1.0, 1.0, 1.0])
        self.assertEqual(actual['source'].to_list(), ['1', '2', '3'])
        self.assertEqual(actual['target'].to_list(), ['2', '3', '4'])

    def test__compute_hidden_network_hashtags_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet._compute_hidden_network(self.tmp_dataset, hashtags=['#a'])

        self.assertEqual({'1', '2', '3', '4'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2'), ('2', '3'), ('3', '4')}, edges)

    def test_get_hidden_network_hashtags_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet.get_hidden_network(self.tmp_dataset, hashtags=['#a'])

        self.assertEqual({'1', '2', '3', '4'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2'), ('2', '3'), ('3', '4')}, edges)

    def test_get_egonet_hashtags_filtering(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        user_id = '1'
        depth = 1

        actual = self.egonet.get_egonet(self.tmp_dataset, user_id, depth, hashtags=['#a'])

        self.assertEqual({'1', '2'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2')}, edges)

    def test_get_egonet_hashtags_filtering_missing(self):
        # Test the case where the user is not present in the filtered dataset by hashtag
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        user_id = '1'
        depth = 1

        # assert it raises
        with self.assertRaises(KeyError):
            actual = self.egonet.get_egonet(self.tmp_dataset, user_id, depth, hashtags=['#e'])

    def test_get_egonet_hashtags_filtering_missing_2(self):
        # Test the case where the hashtag is not present in the dataset
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        user_id = '1'
        depth = 1

        with self.assertRaises(KeyError):
            actual = self.egonet.get_egonet(self.tmp_dataset, user_id, depth, hashtags=['#e'])

    def test_get_hidden_network_hashtags_filtering_backbone(self):
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4'],
                                       'hashtags': [['#a', '#b'], ['#a', '#c'], ['#a', '#d'], ['#b', '#c'],
                                                    ['#b', '#d']]})
        test_data = create_test_data_from_edges_with_hashtags(expected_edges)

        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)

        actual = self.egonet.get_hidden_network_backbone(self.tmp_dataset, hashtags=['#a'])

        self.assertEqual({'1', '2'}, set(actual.vs['author_id']))
        edges = {(actual.vs[s]['author_id'], actual.vs[t]['author_id']) for s, t in actual.get_edgelist()}
        self.assertEqual({('1', '2')}, edges)

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
        self.egonet.get_hidden_network = Mock(return_value=self.egonet._compute_hidden_network(self.test_dataset))

        actual = self.egonet._compute_hidden_network_backbone(self.test_dataset)

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
        end_time = time.time()
        print(f'loaded in {end_time - start_time} seconds')
        start_time = time.time()

        expected_hidden_network = self.egonet._compute_hidden_network(self.tmp_dataset)
        expected_hidden_network_backbone = self.egonet.compute_backbone(expected_hidden_network,
                                                                        alpha=self.egonet.threshold)

        end_time = time.time()
        print(f'computed in {end_time - start_time} seconds')

        actual_hidden_network = self.egonet.get_hidden_network(self.tmp_dataset)
        actual_hidden_network_backbone = self.egonet.get_hidden_network_backbone(self.tmp_dataset)

        self.assertEqual(expected_hidden_network.vcount(), actual_hidden_network.vcount())
        self.assertEqual(expected_hidden_network.ecount(), actual_hidden_network.ecount())
        self.assertEqual(expected_hidden_network_backbone.vcount(), actual_hidden_network_backbone.vcount())
        self.assertEqual(expected_hidden_network_backbone.ecount(), actual_hidden_network_backbone.ecount())

    # Test dataset

    def test_compute_hidden_network_full(self):
        actual = self.egonet._compute_hidden_network(self.test_dataset)

        self.assertEqual(3321, actual.vcount())
        self.assertEqual(5847, actual.ecount())

    def test_compute_hidden_network_full_no_unknown_authors(self):
        self.egonet.include_unknown_authors = False
        actual = self.egonet._compute_hidden_network(self.test_dataset)

        self.assertEqual(3321, actual.vcount())
        self.assertEqual(5847, actual.ecount())

    def test_compute_hidden_network_weight_full(self):
        graph = self.egonet.get_hidden_network(self.test_dataset)

        actual = pd.Series(graph.es['weight']).value_counts()[:5].to_list()
        expected = [4935, 430, 136, 99, 49]
        self.assertEqual(expected, actual)

        actual = pd.Series(graph.es['weight_inv']).value_counts().to_list()[:5]
        expected = [4935, 430, 136, 99, 49]
        self.assertEqual(expected, actual)

        actual = pd.Series(graph.es['weight_norm']).value_counts().to_list()[:5]
        expected = [2404, 157, 139, 121, 88]
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

    def test__get_references_full(self):
        actual = self.egonet._get_references(self.test_dataset)
        actual = actual.sort_values(['source', 'target']).reset_index(drop=True)
        expected = [4935, 430, 136, 99, 49, 30, 28, 19, 15, 15, 14, 12, 10, 9, 7, 7, 7, 6, 5, 4, 3, 3, 2, 1, 1]
        expected_weight_norm = [2404, 157, 139, 121, 88, 87, 78, 77, 74, 72]
        self.assertEqual(expected, actual['weight'].value_counts().to_list(), )
        self.assertEqual(expected, actual['weight_inv'].value_counts().to_list(), expected)
        self.assertEqual(expected_weight_norm, actual['weight_norm'].value_counts().to_list()[:10], )
        self.assertEqual(['1000010057743065089', '1000300778190528512'], actual['source'].to_list()[:2])
        self.assertEqual(['270839361', '420374996'], actual['target'].to_list()[:2])

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
        self.assertEqual(2528, backbone.vcount())
        self.assertEqual(2369, backbone.ecount())

    def test_backbone_full_nothing(self):
        network = self.egonet._compute_hidden_network(self.test_dataset)
        backbone = self.egonet.compute_backbone(network, alpha=1)
        self.assertEqual(backbone.vcount(), 0)
        self.assertEqual(backbone.ecount(), 0)

    def _test_show_alpha_distribution(self):
        network = self.egonet._compute_hidden_network(self.test_dataset)

        alphas = self.egonet.compute_alphas(network)
        # plot alphas histogram with plotly
        fig = px.histogram(alphas, nbins=1000)
        fig.update_xaxes(title_text='Alpha')
        fig.update_yaxes(title_text='Count')
        # fig.show()

    def test_persistence_and_loading_full(self):
        # Test the persistence and loading of the graph
        self.egonet.persist([self.test_dataset])

        start_time = time.time()
        end_time = time.time()
        print(f'loaded in {end_time - start_time} seconds')
        start_time = time.time()

        expected_hidden_network = self.egonet._compute_hidden_network(self.test_dataset)
        expected_hidden_network_backbone = self.egonet.compute_backbone(expected_hidden_network,
                                                                        alpha=self.egonet.threshold)

        end_time = time.time()
        print(f'computed in {end_time - start_time} seconds')

        actual_hidden_network = self.egonet.get_hidden_network(self.test_dataset)
        actual_hidden_network_backbone = self.egonet.get_hidden_network_backbone(self.test_dataset)

        self.assertEqual(expected_hidden_network.vcount(), actual_hidden_network.vcount())
        self.assertEqual(expected_hidden_network.ecount(), actual_hidden_network.ecount())
        self.assertEqual(expected_hidden_network_backbone.vcount(), actual_hidden_network_backbone.vcount())
        self.assertEqual(expected_hidden_network_backbone.ecount(), actual_hidden_network_backbone.ecount())


if __name__ == '__main__':
    unittest.main()
