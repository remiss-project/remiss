import random
import unittest
import uuid

import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timestamp
from pymongo import MongoClient
from tqdm import tqdm

from propagation import DiffusionMetrics

import igraph as ig


class DiffusionMetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.diffusion_metrics = DiffusionMetrics()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)
        self.test_tweet_id = '1167078759280889856'
        self.missing_tweet_id = '1077146799692021761'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_plot_propagation_lengths(self):
        df = self.diffusion_metrics.get_conversation_sizes(self.test_dataset)
        print(df.head(20))
        df.hist(log=True, bins=100)
        for tweet_id in tqdm(df.index[:20]):
            try:
                conversation_id, tweets, references = self.diffusion_metrics.get_conversation(self.test_dataset,
                                                                                              self.test_tweet_id)
                if len(references) > 0:
                    print(f'{tweet_id}: {len(references)}', df.loc[tweet_id])
                    break
            except Exception as e:
                pass
        plt.show()

    def test_propagation_tree(self):
        graph = self.diffusion_metrics.get_propagation_tree(self.test_dataset, self.test_tweet_id)
        self.assertEqual(graph.vcount(), 8)
        self.assertEqual(graph.ecount(), 7)
        self.assertEqual(graph.is_directed(), True)
        self.assertIsInstance(graph.vs['author_id'][0], str)
        self.assertIsInstance(graph.vs['created_at'][0], Timestamp)

        # Display the igraph graph with matplotlib
        layout = graph.layout('fr')
        fig, ax = plt.subplots()
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

    def test_all_connected_to_conversation_id(self):
        graph = self.diffusion_metrics.get_propagation_tree(self.test_dataset, self.test_tweet_id)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_tweets_with_references(self):
        conversation_id, tweets, references = self.diffusion_metrics.get_conversation(self.test_dataset,
                                                                                      self.test_tweet_id)

        self.assertEqual(61, len(references), )
        self.assertEqual(76, len(tweets), )
        self.assertEqual(tweets.columns.tolist(),
                         ['tweet_id', 'author_id', 'created_at'])

    def test_propagation_tree_simple(self):
        client = MongoClient('localhost', 27017)

        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        original_graph = ig.Graph(n=8, edges=edges, directed=True)
        original_graph.vs['label'] = [str(i) for i in range(8)]
        fig, ax = plt.subplots()
        layout = original_graph.layout('fr')
        ig.plot(original_graph, layout=layout, target=ax)
        timestamps = [Timestamp.now() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}'
                    } for i in range(8)]
        data = [{
            'id': str(i),
            'author': {'id': f'author_id_{i}'},
            'conversation_id': '0',
            'referenced_tweets': [{'type': 'retweeted',
                                   'id': str(j),
                                   'author': authors[j],
                                   'created_at': timestamps[j]}],
            'created_at': timestamps[i]

        } for i, j in edges]
        collection.insert_many(data)
        graph = self.diffusion_metrics.get_propagation_tree(self.tmp_dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set(
            (graph.vs['author_id'][edge.source], graph.vs['author_id'][edge.target]) for edge in graph.es)
        expected_edges = {(f'author_id_{source}', f'author_id_{target}') for source, target in edges}
        self.assertEqual(actual_edges, expected_edges)

    def test_propagation_tree_disconnected(self):
        client = MongoClient('localhost', 27017)
        dataset = 'test_simple_dataset'
        client.drop_database(dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        edges = [(0, 1), (1, 2), (2, 4), (2, 5), (3, 6), (3, 7)]
        original_graph = ig.Graph(n=8, edges=edges, directed=True)
        original_graph.vs['label'] = [str(i) for i in range(8)]
        fig, ax = plt.subplots()
        layout = original_graph.layout('fr')
        ig.plot(original_graph, layout=layout, target=ax)
        timestamps = [Timestamp.now().date() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    'remiss_metadata': {
                        'is_usual_suspect': random.choice([True, False]),
                        'party': random.choice([None, 'party1', 'party2']),
                    }
                    } for i in range(8)]
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        test_data = []
        for source, targets in edges_df.groupby('source'):
            test_data.append({
                'id': str(source),
                'conversation_id': '0',
                'referenced_tweets': [{'type': 'retweeted',
                                       'id': str(target),
                                       'author': authors[target],
                                       'created_at': timestamps[target]} for target in targets['target']],
                'username': f'username_{source}',
                'author': authors[source],
                'created_at': timestamps[source]
            })
        collection.insert_many(test_data)
        references_created_at = []
        for tweet in test_data:
            for reference in tweet['referenced_tweets']:
                references_created_at.append((reference['id'], reference['created_at']))

        references_created_at = pd.DataFrame(references_created_at, columns=['id', 'created_at'])

        graph = self.diffusion_metrics.get_propagation_tree(dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set((graph.vs['author_id'][edge.source], graph.vs['author_id'][edge.target]) for edge in graph.es)
        expected_edges = {(f'author_id_{source}', f'author_id_{target}') for source, target in edges}
        expected_edges.add(('author_id_0', 'author_id_3'))
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)

        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_propagation_tree_missing_conversation_id(self):
        client = MongoClient('localhost', 27017)
        dataset = 'test_simple_dataset'
        client.drop_database(dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        original_graph = ig.Graph(n=8, edges=edges, directed=True)
        original_graph.vs['label'] = [str(i) for i in range(8)]
        fig, ax = plt.subplots()
        layout = original_graph.layout(self.diffusion_metrics.layout)
        ig.plot(original_graph, layout=layout, target=ax)
        timestamps = [Timestamp.now().date() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    'remiss_metadata': {
                        'is_usual_suspect': random.choice([True, False]),
                        'party': random.choice([None, 'party1', 'party2']),
                    }
                    } for i in range(8)]
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        test_data = []
        for source, targets in edges_df.groupby('source'):
            test_data.append({
                'id': str(source),
                'conversation_id': '0',
                'referenced_tweets': [{'type': 'retweeted',
                                       'id': str(target),
                                       'author': authors[target],
                                       'created_at': timestamps[target]} for target in targets['target']],
                'username': f'username_{source}',
                'author': authors[source],
                'created_at': timestamps[source]
            })
        collection.insert_many(test_data)
        references_created_at = []
        for tweet in test_data:
            for reference in tweet['referenced_tweets']:
                references_created_at.append((reference['id'], reference['created_at']))

        references_created_at = pd.DataFrame(references_created_at, columns=['id', 'created_at'])

        graph = self.diffusion_metrics.get_propagation_tree(dataset, '1')
        fig, ax = plt.subplots()
        layout = graph.layout(self.diffusion_metrics.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = pd.DataFrame(
            [(graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es],
            columns=['source', 'target'])
        actual_edges['source'] = actual_edges['source'].str.replace('-', 'username_0').astype(str)
        actual_edges['target'] = actual_edges['target'].str.replace('-', 'username_0').astype(str)
        actual_edges = set(actual_edges.itertuples(index=False, name=None))
        edges.append((0, 1))
        expected_edges = {(f'username_{source}', f'username_{target}') for source, target in edges}
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)

        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_conversation_no_nat(self):
        conversation_id, tweets, references = self.diffusion_metrics.get_conversation(self.test_dataset,
                                                                                      self.test_tweet_id)
        self.assertFalse(tweets.isna().any().any())
        self.assertFalse(references.isna().any().any())

    def test_depth_plot(self):
        df = self.diffusion_metrics.get_depth_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 8)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 1)

    def test_size_plot(self):
        df = self.diffusion_metrics.get_size_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 8)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 8)

    def test_max_breadth_plot(self):
        df = self.diffusion_metrics.get_max_breadth_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 8)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 7)

    def test_structured_virality_plot(self):
        df = self.diffusion_metrics.get_structural_virality_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 8)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 1.53125)

    def _test_depth_cascade_ccdf_plot(self):
        start_time = Timestamp.now()
        df = self.diffusion_metrics.get_depth_cascade_ccdf(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')

    def test_size_cascade_ccdf_plot(self):
        start_time = Timestamp.now()
        df = self.diffusion_metrics.get_size_cascade_ccdf(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        self.assertEqual(df.columns.to_list(), ['Normal', 'Politician'])

    def test_cascade_count_over_time_plot(self):
        start_time = Timestamp.now()
        df = self.diffusion_metrics.get_cascade_count_over_time(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        self.assertEqual('Cascade Count', df.name)


if __name__ == '__main__':
    unittest.main()