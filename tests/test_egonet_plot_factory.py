import random
import random
import shutil
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import igraph as ig
import numpy as np
import pandas as pd
from pymongo import MongoClient

from figures import EgonetPlotFactory
from figures.egonet import compute_backbone


class TestEgonetPlotFactory(unittest.TestCase):
    def setUp(self):
        self.egonet_plot = EgonetPlotFactory()
        self.database = 'test_remiss'
        self.host = 'localhost'
        self.port = 27017
        self.egonet_plot.database = self.database
        self.egonet_plot.host = self.host
        self.egonet_plot.port = self.port
        self.collection = 'test_collection'

    def tearDown(self) -> None:
        client = MongoClient('localhost', 27017)
        client.drop_database(self.database)
        if self.egonet_plot.cache_dir:
            shutil.rmtree(self.egonet_plot.cache_dir)

    @patch('figures.MongoClient')
    def test_get_egonet(self, mock_mongo_client):
        # Checks it returns the whole thing if the user is not present

        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, False, True],
                                        'party': ['PSOE', None, 'VOX', None],
                                        'username': ['TEST_USER_1', 'TEST_USER_2', 'TEST_USER_3', 'TEST_USER_4']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'
        user = 'TEST_USER_1'
        depth = 1

        actual = self.egonet_plot.get_egonet(collection, user, depth)

        self.assertEqual({1, 2}, set(actual.vs['id_']))
        edges = {(actual.vs[s]['id_'], actual.vs[t]['id_']) for s, t in actual.get_edgelist()}
        self.assertEqual({(1, 2)}, edges)

    @patch('figures.MongoClient')
    def test_get_egonet_2(self, mock_mongo_client):
        # Checks it returns the whole thing if the user is not present

        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, False, True],
                                        'party': ['PSOE', None, 'VOX', None],
                                        'username': ['TEST_USER_1', 'TEST_USER_2', 'TEST_USER_3', 'TEST_USER_4']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'
        user = 'TEST_USER_4'
        depth = 1

        actual = self.egonet_plot.get_egonet(collection, user, depth)

        self.assertEqual({1, 2}, set(actual.vs['id_']))
        edges = {(actual.vs[s]['id_'], actual.vs[t]['id_']) for s, t in actual.get_edgelist()}
        self.assertEqual({(3, 4)}, edges)

    @patch('figures.MongoClient')
    def test_compute_hidden_network(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 1],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3],
                                        'is_usual_suspect': [False, False, False],
                                        'party': ['PSOE', None, 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        # Mock get_user_id
        self.egonet_plot.get_user_id = Mock(return_value=1)
        collection = 'test_collection'

        actual = self.egonet_plot.get_hidden_network(collection)

        self.assertEqual({1, 2, 3}, set(actual.vs['id_']))
        edges = {frozenset((actual.vs[s]['id_'], actual.vs[t]['id_'])) for s, t in actual.get_edgelist()}
        expected = {(2, 3), (1, 2), (1, 3)}
        expected = {frozenset(x) for x in expected}
        self.assertEqual(expected, edges)

    @patch('figures.MongoClient')
    def test_compute_hidden_network_2(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, False, False],
                                        'party': ['PSOE', None, 'VOX', None],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        # Mock get_user_id
        self.egonet_plot.get_user_id = Mock(return_value=1)
        collection = 'test_collection'

        actual = self.egonet_plot.get_hidden_network(collection)

        self.assertEqual({1, 2, 3, 4}, set(actual.vs['id_']))
        edges = {(actual.vs[s]['id_'], actual.vs[t]['id_']) for s, t in actual.get_edgelist()}
        self.assertEqual({(2, 3), (1, 2), (3, 4)}, edges)

    @patch('figures.MongoClient')
    def test_compute_hidden_network_weight(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, False, False],
                                        'party': ['PSOE', None, 'VOX', None],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        # Mock get_user_id
        self.egonet_plot.get_user_id = Mock(return_value=1)
        collection = 'test_collection'

        graph = self.egonet_plot.get_hidden_network(collection)

        actual = graph.es['weight']
        expected = [1, 2, 3]
        self.assertEqual(expected, actual)

        actual = graph.es['weight_inv']
        expected = [1, 0.5, 0.33]
        self.assertEqual(expected, actual)

        actual = graph.es['weight_norm']
        expected = [1, 0.5, 0.5]
        self.assertEqual(expected, actual)

    def test__get_references(self):
        test_data = []
        expected_edges = pd.DataFrame({'source': [1, 2, 3, 2, 1],
                                       'target': [2, 3, 4, 3, 4]})
        for source, target in expected_edges.itertuples(index=False):
            party = random.choice(['PSOE', 'PP', 'VOX', 'UP', None])
            is_usual_suspect = random.choice([True, False])
            referenced_tweets = [
                {"id": f'{source}->{target}', "author": {"id": target, "username": f"TEST_USER_{target}"},
                 "type": "retweeted"}]
            tweet = {"id": f'{source}->{target}', "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{source}", "id": source,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            test_data.append(tweet)

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        collection.insert_many(test_data)

        # Mock get_user_id
        self.egonet_plot.get_user_id = Mock(return_value=1)
        collection = 'test_collection'

        actual = self.egonet_plot._get_references(collection)
        actual = actual.sort_values(['source', 'target']).reset_index(drop=True)
        self.assertEqual(actual['weight'].to_list(), [1, 1, 2, 1])
        self.assertEqual(actual['weight_inv'].to_list(), [1, 1, 0.5, 1])
        self.assertEqual(actual['weight_norm'].to_list(), [0.5, 0.5, 1, 1])
        self.assertEqual(actual['source'].to_list(), [1, 1, 2, 3])
        self.assertEqual(actual['target'].to_list(), [2, 4, 3, 4])

    def test_plot_egonet(self):
        # Mock get_egonet
        network = ig.Graph.GRG(8, 0.2)
        network.vs['id_'] = [0, 1, 2, 3, 4, 5, 6, 7]
        network.vs['username'] = ['TEST_USER_0', 'TEST_USER_1', 'TEST_USER_2', 'TEST_USER_3', 'TEST_USER_4',
                                  'TEST_USER_5', 'TEST_USER_6', 'TEST_USER_7']
        network.vs['party'] = ['PSOE', None, 'VOX', None, 'PSOE', None, 'VOX', None]
        network.vs['is_usual_suspect'] = [False, False, False, False, True, True, True, True]
        self.egonet_plot.get_egonet = Mock(return_value=network)

        collection = 'test_collection'

        actual = self.egonet_plot.plot_egonet(collection, 'test_user', 1)
        self.assertEqual(len(actual['data'][0]['x']), network.ecount() * 3)
        self.assertEqual(len(actual['data'][0]['y']), network.ecount() * 3)
        self.assertEqual(len(actual['data'][0]['z']), network.ecount() * 3)
        self.assertEqual(len(actual['data'][1]['x']), network.vcount())
        self.assertEqual(len(actual['data'][1]['y']), network.vcount())
        self.assertEqual(len(actual['data'][1]['z']), network.vcount())

    def test_get_authors_and_references(self):
        # Checks it returns the whole thing if the user is not present
        data_size = 100

        collection = 'test_collection'
        authors = self.egonet_plot._get_authors(collection)
        references = self.egonet_plot._get_references(collection)
        self.assertEqual(data_size // 2, len(authors))
        expected_referenced_tweets = expected_references.shape[0] - expected_references[
            ['source', 'target']].duplicated().sum()
        self.assertEqual(len(references), expected_referenced_tweets)
        authors = authors.sort_values('id').reset_index(drop=True)
        expected_authors = expected_authors.sort_values('id').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_authors, authors,
                                      check_dtype=False, check_like=True)
        references = references.sort_values(['source', 'target']).reset_index(drop=True)
        expected_references = expected_references.sort_values(['source', 'target']).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_references, references,
                                      check_dtype=False, check_like=True)

    def test_get_egonet_2(self):
        # Checks it returns the whole thing if the user is not present

        collection = 'test_collection'
        user = 'test_user'
        depth = 1
        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        actual = self.egonet_plot.get_egonet(collection, user, depth)
        self.assertEqual(data_size // 2, actual.vcount())
        self.assertEqual(len(expected_references), actual.ecount())
        actual_authors = pd.DataFrame({'id': actual.vs['id_'],
                                       'username': actual.vs['username'],
                                       'party': actual.vs['party'],
                                       'is_usual_suspect': actual.vs['is_usual_suspect']})
        actual_authors = actual_authors.sort_values('id').reset_index(drop=True)
        expected_authors = expected_authors.sort_values('id').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_authors, actual_authors,
                                      check_dtype=False, check_like=True)
        actual_references = {frozenset([actual.vs[s]['id_'], actual.vs[t]['id_']]) for s, t in actual.get_edgelist()}
        expected_references = {frozenset([x['source'], x['target']]) for _, x in expected_references.iterrows()}
        self.assertEqual(expected_references, actual_references)

    def test_get_egonet_speed(self):
        # Checks it returns the whole thing if the user is not present

        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        # time computation of get_egonet
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLessEqual(total_time, 10)

    def test_cache(self):
        # Checks it returns the whole thing if the user is not present
        user = 'test_user'
        depth = 1
        test_data = create_test_data_db(self.database, self.collection)
        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        self.egonet_plot.cache_dir = Path('/tmp/remiss_cache')
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(self.collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        self.assertLessEqual(total_time_no_cache, 60)
        self.assertTrue(Path(f'/tmp/remiss_cache/{self.collection}/hidden_network.graphmlz').exists())
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(self.collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)
        self.assertTrue(actual.get_edgelist(),
                        ig.Graph.Read_GraphMLz(
                            f'/tmp/remiss_cache/{self.collection}/hidden_network.graphmlz').get_edgelist())
        self.assertEquals(len(actual['layout'].coords), actual.vcount())

    def test_cache_simplified_hidden_untouched(self):
        # Checks it returns the whole thing if the user is not present
        user = 'test_user'
        depth = 1
        test_data = create_test_data_db(self.database, self.collection)
        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        self.egonet_plot.cache_dir = Path('/tmp/remiss_cache')
        self.egonet_plot.simplification = 'backbone'
        self.egonet_plot.threshold = 0.7
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(self.collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        self.assertLessEqual(total_time_no_cache, 60)
        self.assertTrue(Path(f'/tmp/remiss_cache/{self.collection}/hidden_network.graphmlz').exists())
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(self.collection, user, depth)
        actual = self.egonet_plot._hidden_networks[self.collection]
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)
        self.assertTrue(actual.get_edgelist(),
                        ig.Graph.Read_GraphMLz(
                            f'/tmp/remiss_cache/{self.collection}/hidden_network.graphmlz').get_edgelist())
        self.assertEquals(len(actual['layout'].coords), actual.vcount())


    def test_backbone(self):
        test_data = []
        expected_edges = pd.DataFrame({'source': [1, 2, 2, 1, 1, 1, 1, 1],
                                       'target': [2, 3, 3, 2, 2, 2, 2, 4]})
        for source, target in expected_edges.itertuples(index=False):
            party = random.choice(['PSOE', 'PP', 'VOX', 'UP', None])
            is_usual_suspect = random.choice([True, False])
            referenced_tweets = [
                {"id": f'{source}->{target}', "author": {"id": f'{target}', "username": f"TEST_USER_{target}"},
                 "type": "retweeted"}]
            tweet = {"id": f'{source}->{target}', "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{source}", "id": f'{source}',
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            test_data.append(tweet)

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        collection.insert_many(test_data)

        collection = 'test_collection'
        network = self.egonet_plot._compute_hidden_network(collection)
        backbone = compute_backbone(network, alpha=0.2)
        actual = {frozenset(x) for x in backbone.get_edgelist()}
        expected = {frozenset((0, 1))}
        self.assertEqual(expected, actual)

    def test_backbone_2(self):
        # Create a test graph
        alpha = 0.05

        network = ig.Graph.Erdos_Renyi(250, 0.02, directed=False)
        network.es["weight_norm"] = np.random.uniform(0, 0.5, network.ecount())

        # Test the backbone filter
        backbone = compute_backbone(network, alpha=alpha)

        # Ensure that all edge weights are below alpha
        for edge in backbone.get_edgelist():
            weight = backbone.es[backbone.get_eid(*edge)]['weight_norm']
            degree = backbone.degree(edge[0])
            edge_alpha = (1 - weight) ** (degree - 1)
            self.assertGreater(edge_alpha, alpha)

    def test_cache_backbone(self):
        # Checks it returns the whole thing if the user is not present
        self.egonet_plot.simplification = 'backbone'
        self.egonet_plot.threshold = 0.4
        user = 'test_user'
        depth = 1
        test_data = create_test_data_db(self.database, self.collection)

        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        self.egonet_plot.cache_dir = Path('/tmp/remiss_cache')
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(self.collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        self.assertLessEqual(total_time_no_cache, 60)
        self.assertTrue(Path(f'/tmp/remiss_cache/{self.collection}/hidden_network-backbone-0.4.graphmlz').exists())
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(self.collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)
        self.assertTrue(actual.get_edgelist(),
                        ig.Graph.Read_GraphMLz(
                            f'/tmp/remiss_cache/{self.collection}/hidden_network-backbone-0.4.graphmlz').get_edgelist())
        self.assertEquals(len(actual['layout'].coords), actual.vcount())

    def test_plot_cache(self):
        # Checks it returns the whole thing if the user is not present
        self.egonet_plot.simplification = 'backbone'
        self.egonet_plot.threshold = 0.4

        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        self.egonet_plot.cache_dir = Path('/tmp/remiss_cache')
        start_time = time.time()
        actual = self.egonet_plot.plot_egonet(collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        start_time = time.time()
        actual = self.egonet_plot.plot_egonet(collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)

    @patch('figures.MongoClient')
    def test_check_color_coding(self, mock_mongo_client):

        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, True, True],
                                        'party': [None, 'PSOE', None, 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected_colors = ['blue', 'yellow', 'red', 'purple']

        plot = self.egonet_plot.plot_egonet(collection, None, 1)
        self.assertEqual(list(plot['data'][1]['marker']['color']), expected_colors)

    @patch('figures.MongoClient')
    def test_check_color_coding_2(self, mock_mongo_client):

        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, True, True],
                                        'party': [None, None, None, 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected_colors = ['blue', 'blue', 'red', 'purple']

        plot = self.egonet_plot.plot_egonet(collection, None, 1)
        self.assertEqual(list(plot['data'][1]['marker']['color']), expected_colors)

    @patch('figures.MongoClient')
    def test_check_color_coding_3(self, mock_mongo_client):

        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [False, False, True, True],
                                        'party': ['PSOE', 'PSOE', None, 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected_colors = ['yellow', 'yellow', 'red', 'purple']

        plot = self.egonet_plot.plot_egonet(collection, None, 1)
        self.assertEqual(list(plot['data'][1]['marker']['color']), expected_colors)

    @patch('figures.MongoClient')
    def test_check_color_coding_4(self, mock_mongo_client):

        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if 'source' in pipeline[-1]['$project']:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4],
                                      'weight': [1, 2, 3],
                                      'weight_inv': [1, 0.5, 0.33],
                                      'weight_norm': [1, 0.5, 0.5]})
                return edges
            else:
                # its authors
                authors = pd.DataFrame({'id': [1, 2, 3, 4],
                                        'is_usual_suspect': [True, True, True, True],
                                        'party': ['VOX', 'VOX', 'VOX', 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683']})
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected_colors = ['purple', 'purple', 'purple', 'purple']

        plot = self.egonet_plot.plot_egonet(collection, None, 1)
        self.assertEqual(list(plot['data'][1]['marker']['color']), expected_colors)

    def test_no_isolated_vertices(self):
        # Checks it returns the whole thing if the user is not present
        self.egonet_plot.simplification = 'backbone'
        self.egonet_plot.threshold = 0.232
        self.egonet_plot.cache_dir = None
        data_size = 10000
        max_num_references = 10

        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        actual = self.egonet_plot.plot_egonet(collection, user, depth)

        # Check that every vertex is connected to at least one edge
        vertices = actual['data'][1]
        vertices = np.vstack((vertices['x'], vertices['y'])).T
        edges = actual['data'][0]
        edges = np.vstack((edges['x'], edges['y'])).T
        vertices = {tuple(coord) for coord in vertices if not np.isnan(coord).all()}
        edges = {tuple(coord) for coord in edges if not np.isnan(coord).all()}

        self.assertEqual(edges, vertices)

    def test_legitimacy(self):

        expected_legitimacy['id'] = expected_legitimacy['username'].str.replace('TEST_USER_', '').astype(np.int32)
        expected_legitimacy = expected_legitimacy.set_index('id')
        # expected_legitimacy = expected_legitimacy / expected_legitimacy.max()
        expected_legitimacy = expected_legitimacy.sort_values('legitimacy', ascending=False)
        pd.testing.assert_frame_equal(expected_legitimacy, actual, check_dtype=False, check_like=True)

    def test_reputation(self):
        # compute reputation as the amount of referenced tweets attained by each user
        data_size = 100
        day_range = 10
        max_num_references = 20

        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        actual = self.egonet_plot.get_reputation(collection)
        expected = pd.DataFrame({'username': [t['author']['username'] for t in test_data],
                                 'id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['id', 'username', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index=['id', 'username'], columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.cumsum(axis=1)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)

    def test__get_legitimacy_per_time_1(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data_db(database=self.database, collection=self.collection,
                                        day_range=day_range)

        self.egonet_plot.unit = 'day'
        self.egonet_plot.bin_size = 20 + day_range
        actual = self.egonet_plot._get_legitimacy_per_time(self.collection)
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id'])['legitimacy'].sum()

        expected = expected.to_frame()
        expected.columns = actual.columns
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False)

    def test__get_legitimacy_per_time_2(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data_db(database=self.database, collection=self.collection,
                                        day_range=day_range)
        self.egonet_plot.database = 'test_remiss'
        actual = self.egonet_plot._get_legitimacy_per_time(self.collection)
        expected = pd.DataFrame({
            'author_id': [t['author']['id'] for t in test_data],
            'date': [t['created_at'].date() for t in test_data],
            'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)

    def test_status(self):
        # compute status as the amount of referenced tweets attained by each user
        day_range = 10
        test_data = create_test_data_db(database=self.database, collection=self.collection,
                                        day_range=day_range)
        actual = self.egonet_plot.get_status(self.collection)
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.cumsum(axis=1)
        expected = expected.apply(lambda x: x.argsort())
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False)

    def test_egonet_date(self):
        data_size = 100
        day_range = 10
        max_num_references = 20
        start_date = datetime.fromisoformat("2019-01-01T00:00:00Z")
        end_date = start_date
        test_data = create_test_data_db(database=self.database, collection=self.collection,
                                        day_range=day_range, data_size=data_size, max_num_references=max_num_references)

        authors = self.egonet_plot._get_authors(self.collection, start_date=start_date, end_date=end_date)
        references = self.egonet_plot._get_references(self.collection, start_date=start_date, end_date=end_date)
        self.assertEqual(len(authors), 0)
        self.assertEqual(len(references), 0)

    def test_egonet_date_2(self):
        data_size = 100
        day_range = 10
        max_num_references = 20
        start_date = datetime.fromisoformat("2019-01-01T00:00:00Z")
        end_date = datetime.fromisoformat("2019-01-01T00:00:00Z") + timedelta(days=1)

        test_data = create_test_data_db(database=self.database, collection=self.collection,
                                        day_range=day_range, data_size=data_size, max_num_references=max_num_references)
        authors = self.egonet_plot._get_authors(self.collection, start_date=start_date, end_date=end_date)
        authors = authors.sort_values('author_id').reset_index(drop=True)
        expected_authors = {}
        expected_references = []
        for tweet in test_data:
            if tweet['created_at'] <= start_date or tweet['created_at'] < end_date:
                expected_authors[tweet['author']['id']] = [tweet['author']['id'], tweet['author']['username'],
                                                           tweet['author']['remiss_metadata']['is_usual_suspect'],
                                                           tweet['author']['remiss_metadata']['party']]
                for referenced_tweet in tweet['referenced_tweets']:
                    expected_references.append([tweet['author']['id'], referenced_tweet['author']['id']])

        expected_authors = pd.DataFrame(expected_authors.values(),
                                        columns=['author_id', 'username', 'is_usual_suspect', 'party'])
        expected_authors = expected_authors.drop_duplicates().sort_values('author_id').reset_index(drop=True)
        authors = authors.sort_values('author_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_authors, authors, check_dtype=False, check_like=True,
                                      check_index_type=False)

        references = self.egonet_plot._get_references(self.collection, start_date=start_date, end_date=end_date)
        references = references.sort_values(['source', 'target']).reset_index(drop=True)
        references = references[['source', 'target']]
        expected_references = pd.DataFrame(expected_references, columns=['source', 'target'])
        expected_references = expected_references.drop_duplicates().sort_values(['source', 'target']).reset_index(
            drop=True)
        pd.testing.assert_frame_equal(expected_references, references, check_dtype=False, check_like=True,
                                      check_index_type=False)


def create_test_data_db(database='unit_test_remiss', collection='unit_test', data_size=100, day_range=10,
                        max_num_references=20):
    test_data = []
    total_referenced_tweets = 0
    usual_suspects = {}
    parties = {}
    for i in range(data_size):
        for day in range(day_range):
            if i // 2 not in usual_suspects:
                usual_suspects[i // 2] = random.choice([True, False])
            if i // 2 not in parties:
                parties[i // 2] = random.choice(['PSOE', 'PP', 'VOX', 'UP', None])

            num_referenced_tweets = random.randint(0, max_num_references)
            total_referenced_tweets += num_referenced_tweets
            referenced_tweets = []
            for j in range(num_referenced_tweets):
                author_id = random.randint(0, data_size // 2 - 1)
                referenced_tweets.append(
                    {'id': i + 1, 'author': {'id': f'{author_id}', 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            created_at = datetime.fromisoformat("2019-01-01T00:01:00Z") + timedelta(days=day)
            tweet = {"id": i,
                     "author": {"username": f"TEST_USER_{i // 2}", "id": f'{i // 2}',
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets,
                     'created_at': created_at}
            test_data.append(tweet)

    client = MongoClient('localhost', 27017)
    client.drop_database(database)
    database = client.get_database(database)
    collection = database.get_collection(collection)
    collection.insert_many(test_data)

    return test_data
