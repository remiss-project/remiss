import random
import time
import unittest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

import igraph as ig
import numpy as np
import pandas as pd
from pymongo import MongoClient

from figures.propagation import compute_backbone
from propagation import Egonet


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

    def test_get_authors_and_references(self):
        # Checks it returns the whole thing if the user is not present
        data_size = 100
        expected_edges = pd.DataFrame({'source': ['1', '2', '3', '2', '1'],
                                       'target': ['2', '3', '4', '3', '4']})
        test_data = create_test_data_from_edges(expected_edges)
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        collection.insert_many(test_data)
        authors = self.egonet._get_authors(self.tmp_dataset)
        self.assertEqual({'1', '2', '3', '4'}, set(authors['author_id']))

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
        backbone = compute_backbone(network, alpha=alpha)

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


def create_test_data_from_edges(expected_edges):
    test_data = []
    for source, target in expected_edges.itertuples(index=False):
        party = random.choice(['PSOE', 'PP', 'VOX', 'UP', None])
        is_usual_suspect = random.choice([True, False])
        referenced_tweets = [
            {"id": f'{source}->{target}', "author": {"id": str(target), "username": f"TEST_USER_{target}"},
             "type": "retweeted"}]
        tweet = {"id": f'{source}->{target}', "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                 "author": {"username": f"TEST_USER_{source}", "id": source,
                            "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect}},
                 "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                 'referenced_tweets': referenced_tweets}
        test_data.append(tweet)
    return test_data


def create_test_data(data_size=100, day_range=10,
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
            tweet = {"id": str(i),
                     "author": {"username": f"TEST_USER_{i // 2}", "id": f'{i // 2}',
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect},
                                'public_metrics': {'tweet_count': 1, 'followers_count': 100, 'following_count': 10}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets,
                     'created_at': created_at}
            test_data.append(tweet)

    return test_data


if __name__ == '__main__':
    unittest.main()
