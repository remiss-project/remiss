import random
import random
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import igraph as ig
import numpy as np
import pandas as pd
from pymongo import MongoClient

from figures import EgonetPlotFactory, compute_backbone


class TestEgonetPlotFactory(unittest.TestCase):
    def setUp(self):
        self.egonet_plot = EgonetPlotFactory()

    def tearDown(self) -> None:
        if self.egonet_plot.cache_dir:
            (self.egonet_plot.cache_dir / 'test_collection.graphmlz').unlink(missing_ok=True)
            (self.egonet_plot.cache_dir / 'test_collection.feather').unlink(missing_ok=True)

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

        actual= self.egonet_plot.get_egonet(collection, user, depth)

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

        actual= self.egonet_plot.get_egonet(collection, user, depth)

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

        actual= self.egonet_plot.get_hidden_network(collection)

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

        actual= self.egonet_plot.get_hidden_network(collection)

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

        test_data = []
        total_referenced_tweets = 0
        usual_suspects = {}
        parties = {}
        expected_authors = {}
        expected_references = []
        for i in range(data_size):
            if i // 2 not in usual_suspects:
                usual_suspects[i // 2] = random.choice([True, False])
            if i // 2 not in parties:
                parties[i // 2] = random.choice(['PSOE', 'PP', 'VOX', 'UP', None])

            num_referenced_tweets = random.randint(0, 100)
            total_referenced_tweets += num_referenced_tweets
            referenced_tweets = []
            for j in range(num_referenced_tweets):
                author_id = random.randint(0, data_size // 2 - 1)
                referenced_tweets.append(
                    {'id': i + 1, 'author': {'id': author_id, 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            tweet = {"id": i, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": i // 2,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            expected_authors[i // 2] = {'id': i // 2, 'username': f'TEST_USER_{i // 2}', 'party': party,
                                        'is_usual_suspect': is_usual_suspect}

            expected_references.extend([(i // 2, x['author']['id']) for x in referenced_tweets])
            test_data.append(tweet)

        expected_authors = pd.DataFrame(expected_authors).T
        expected_authors['id'] = expected_authors['id'].astype(int)
        expected_authors['is_usual_suspect'] = expected_authors['is_usual_suspect'].astype(bool)

        expected_references = pd.DataFrame(expected_references, columns=['source', 'target'])
        expected_references = expected_references[['source', 'target']].value_counts().reset_index().rename(
            columns={'count': 'weight'})
        expected_references['weight_inv'] = 1 / expected_references['weight']
        expected_references['weight_norm'] = expected_references['weight'].groupby(
            expected_references['source']).transform(
            lambda x: x / x.sum())

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

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
        data_size = 100
        max_num_references = 1000

        test_data = []
        total_referenced_tweets = 0
        usual_suspects = {}
        parties = {}
        expected_authors = {}
        expected_references = []
        for i in range(data_size):
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
                    {'id': i + 1, 'author': {'id': author_id, 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            tweet = {"id": i, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": i // 2,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            expected_authors[i // 2] = {'id': i // 2, 'username': f'TEST_USER_{i // 2}', 'party': party,
                                        'is_usual_suspect': is_usual_suspect}

            expected_references.extend([(i // 2, x['author']['id']) for x in referenced_tweets])
            test_data.append(tweet)

        expected_authors = pd.DataFrame(expected_authors).T
        expected_authors['id'] = expected_authors['id'].astype(int)
        expected_authors['is_usual_suspect'] = expected_authors['is_usual_suspect'].astype(bool)
        expected_references = pd.DataFrame(expected_references, columns=['source', 'target'])
        expected_references = expected_references[['source', 'target']].value_counts().reset_index().rename(
            columns={'count': 'weight'})
        expected_references['weight_inv'] = 1 / expected_references['weight']

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

        collection = 'test_collection'
        user = 'test_user'
        depth = 1
        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        actual= self.egonet_plot.get_egonet(collection, user, depth)
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
        data_size = 1000
        max_num_references = 1000

        test_data = []
        total_referenced_tweets = 0
        usual_suspects = {}
        parties = {}
        expected_authors = {}
        expected_references = []
        for i in range(data_size):
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
                    {'id': i + 1, 'author': {'id': author_id, 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            tweet = {"id": i, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": i // 2,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            expected_authors[i // 2] = {'id': i // 2, 'username': f'TEST_USER_{i // 2}', 'party': party,
                                        'is_usual_suspect': is_usual_suspect}

            expected_references.extend([(i // 2, x['author']['id']) for x in referenced_tweets])
            test_data.append(tweet)

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

        collection = 'test_collection'
        user = 'test_user'
        depth = 1
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
        data_size = 100
        max_num_references = 100

        test_data = []
        total_referenced_tweets = 0
        usual_suspects = {}
        parties = {}
        expected_authors = {}
        expected_references = []
        for i in range(data_size):
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
                    {'id': i + 1, 'author': {'id': author_id, 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            tweet = {"id": i, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": i // 2,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            expected_authors[i // 2] = {'id': i // 2, 'username': f'TEST_USER_{i // 2}', 'party': party,
                                        'is_usual_suspect': is_usual_suspect}

            expected_references.extend([(i // 2, x['author']['id']) for x in referenced_tweets])
            test_data.append(tweet)

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

        collection = 'test_collection'
        user = 'test_user'
        depth = 1
        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        self.egonet_plot.cache_dir = Path('/tmp/remiss_cache')
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        self.assertLessEqual(total_time_no_cache, 60)
        self.assertTrue(Path('/tmp/remiss_cache/test_collection/hidden_network_graph.graphmlz').exists())
        self.assertTrue(Path('/tmp/remiss_cache/test_collection/hidden_network_layout.feather').exists())
        start_time = time.time()
        actual= self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)
        self.assertTrue(actual.get_edgelist(),
                        ig.Graph.Read_GraphMLz('/tmp/remiss_cache/test_collection/hidden_network_graph.graphmlz').get_edgelist())
        self.assertEqual(actual.vcount(),
                         pd.read_feather('/tmp/remiss_cache/test_collection/hidden_network_layout.feather').shape[0])

    def test_backbone(self):
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

        collection = 'test_collection'
        network = self.egonet_plot._compute_hidden_network(collection)
        backbone = compute_backbone(network, alpha=0.2)
        self.assertEqual(set(backbone.get_edgelist()), {(2, 1), (2, 0)})

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
        data_size = 100
        max_num_references = 100

        test_data = []
        total_referenced_tweets = 0
        usual_suspects = {}
        parties = {}
        expected_authors = {}
        expected_references = []
        for i in range(data_size):
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
                    {'id': i + 1, 'author': {'id': author_id, 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            tweet = {"id": i, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": i // 2,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            expected_authors[i // 2] = {'id': i // 2, 'username': f'TEST_USER_{i // 2}', 'party': party,
                                        'is_usual_suspect': is_usual_suspect}

            expected_references.extend([(i // 2, x['author']['id']) for x in referenced_tweets])
            test_data.append(tweet)

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

        collection = 'test_collection'
        user = 'test_user'
        depth = 1
        print('computing egonet')
        self.egonet_plot.host = 'localhost'
        self.egonet_plot.port = 27017
        self.egonet_plot.database = 'test_remiss'
        self.egonet_plot.cache_dir = Path('/tmp/remiss_cache')
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        self.assertLessEqual(total_time_no_cache, 60)
        self.assertTrue(Path('/tmp/remiss_cache/test_collection/hidden_network_layout-backbone-0.4.feather').exists())
        self.assertTrue(Path('/tmp/remiss_cache/test_collection/hidden_network_graph-backbone-0.4.graphmlz').exists())
        start_time = time.time()
        actual = self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)
        self.assertTrue(actual.get_edgelist(),
                        ig.Graph.Read_GraphMLz('/tmp/remiss_cache/test_collection/hidden_network_graph-backbone-0.4.graphmlz').get_edgelist())
        self.assertEqual(actual.vcount(),
                         pd.read_feather('/tmp/remiss_cache/test_collection/hidden_network_layout-backbone-0.4.feather').shape[0])

    def test_plot_cache(self):
        # Checks it returns the whole thing if the user is not present
        self.egonet_plot.simplification = 'backbone'
        self.egonet_plot.threshold = 0.4
        data_size = 100
        max_num_references = 100

        test_data = []
        total_referenced_tweets = 0
        usual_suspects = {}
        parties = {}
        expected_authors = {}
        expected_references = []
        for i in range(data_size):
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
                    {'id': i + 1, 'author': {'id': author_id, 'username': f'TEST_USER_{author_id}'},
                     'type': 'retweeted'})

            is_usual_suspect = usual_suspects[i // 2]
            party = parties[i // 2]
            tweet = {"id": i, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": i // 2,
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets}
            expected_authors[i // 2] = {'id': i // 2, 'username': f'TEST_USER_{i // 2}', 'party': party,
                                        'is_usual_suspect': is_usual_suspect}

            expected_references.extend([(i // 2, x['author']['id']) for x in referenced_tweets])
            test_data.append(tweet)

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

        collection = 'test_collection'
        user = 'test_user'
        depth = 1
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
