import random
import unittest

import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Timestamp
from pymongo import MongoClient

from figures.propagation import PropagationPlotFactory
from tests.conftest import populate_test_database, delete_test_database
import igraph as ig


class PropagationTestCase(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     populate_test_database('test_dataset')

    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database('test_dataset')

    def setUp(self):
        self.plot_factory = PropagationPlotFactory(available_datasets=['test_dataset'])

    def test_propagation_tree(self):
        graph = self.plot_factory.get_propagation_tree('test_dataset', '1160842257647493120')
        self.assertEqual(graph.vcount(), 76)
        self.assertEqual(graph.ecount(), 75)
        self.assertEqual(graph.is_directed(), True)
        self.assertIsInstance(graph.vs['author_id'][0], str)
        self.assertIsInstance(graph.vs['username'][0], str)
        self.assertIsInstance(graph.vs['tweet_id'][0], str)
        self.assertEqual(graph.vs['party'][0], None)
        self.assertEqual(graph.vs['is_usual_suspect'][0], False)
        self.assertIsInstance(graph.vs['created_at'][0], Timestamp)

        # Display the igraph graph with matplotlib
        layout = graph.layout(self.plot_factory.layout)
        fig, ax = plt.subplots()
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

    def test_all_connected_to_conversation_id(self):
        graph = self.plot_factory.get_propagation_tree('test_dataset', '1160842257647493120')
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)
        shortest_paths = self.plot_factory.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_propagation_lengths(self):
        df = self.plot_factory.get_conversation_sizes('test_dataset')
        for tweet_id in df['Conversation ID'].iloc[11:]:
            try:
                graph = self.plot_factory.get_propagation_tree('test_dataset', tweet_id)
                layout = graph.layout(self.plot_factory.layout)
                fig, ax = plt.subplots()
                ax.set_title(f'Conversation {tweet_id}')
                ig.plot(graph, layout=layout, target=ax)
                plt.show()
                print(f'Conversation {tweet_id} works')
                break
            except (RuntimeError, KeyError) as ex:
                raise ex
        print(df)
        df.hist(log=True, bins=100)
        plt.show()

    def test_tweets_with_references(self):
        tweets, references = self.plot_factory.get_conversation('test_dataset', '1160842257647493120')
        self.assertEqual(tweets.columns.tolist(), ['author_id', 'username', 'is_usual_suspect', 'party'])
        self.assertEqual(len(references), 67)
        self.assertEqual(len(tweets), 76)

    def test_propagation_tree_simple(self):
        client = MongoClient('localhost', 27017)
        dataset = 'test_simple_dataset'
        client.drop_database(dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        original_graph = ig.Graph(n=8, edges=edges, directed=True)
        original_graph.vs['label'] = [str(i) for i in range(8)]
        fig, ax = plt.subplots()
        layout = original_graph.layout(self.plot_factory.layout)
        ig.plot(original_graph, layout=layout, target=ax)
        timestamps = [Timestamp.now() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    'remiss_metadata': {
                        'is_usual_suspect': random.choice([True, False]),
                        'party': random.choice([None, 'party1', 'party2']),
                    }
                    } for i in range(8)]
        collection.insert_many([{
            'id': str(i),
            'conversation_id': '0',
            'referenced_tweets': [{'type': 'replied_to',
                                   'id': str(j),
                                   'author': authors[j],
                                   'created_at': timestamps[j]}],
            'username': f'username_{i}',
            'author': authors[i],
            'created_at': timestamps[i]

        } for i, j in edges])
        graph = self.plot_factory.get_propagation_tree(dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set((graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es)
        self.assertEqual(actual_edges, set(edges))

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
        layout = original_graph.layout(self.plot_factory.layout)
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
                'referenced_tweets': [{'type': 'replied_to',
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

        graph = self.plot_factory.get_propagation_tree(dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set((graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es)
        expected_edges = {(f'username_{source}', f'username_{target}') for source, target in edges}
        expected_edges.add(('username_0', 'username_3'))
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)

        shortest_paths = self.plot_factory.get_shortest_paths_to_conversation_id(graph)
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
        layout = original_graph.layout(self.plot_factory.layout)
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
                'referenced_tweets': [{'type': 'replied_to',
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

        graph = self.plot_factory.get_propagation_tree(dataset, '1')
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = pd.DataFrame([(graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es], columns=['source', 'target'])
        actual_edges['source'] = actual_edges['source'].str.replace('-', 'username_0').astype(str)
        actual_edges['target'] = actual_edges['target'].str.replace('-', 'username_0').astype(str)
        actual_edges = set(actual_edges.itertuples(index=False, name=None))
        edges.append((0, 1))
        expected_edges = {(f'username_{source}', f'username_{target}') for source, target in edges}
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)

        shortest_paths = self.plot_factory.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_propagation_tree_plot(self):
        graph = self.plot_factory.get_propagation_tree('test_dataset', '1160842257647493120')
        fig = self.plot_factory.plot_network(graph)
        fig.show()

    def test_conversation_no_nat(self):
        conversation_id, tweets, references = self.plot_factory.get_conversation('test_dataset', '1160842257647493120')
        tweets = tweets.drop(columns=['party'])
        self.assertFalse(tweets.isna().any().any())
        self.assertFalse(references.isna().any().any())


    def test_depth_plot(self):
        fig = self.plot_factory.plot_depth('test_dataset', '1160842257647493120')
        fig.show()

    def test_size_plot(self):
        fig = self.plot_factory.plot_size('test_dataset', '1160842257647493120')
        fig.show()

    def test_max_breadth_plot(self):
        fig = self.plot_factory.plot_max_breadth('test_dataset', '1160842257647493120')
        fig.show()

    def test_structured_virality_plot(self):
        fig = self.plot_factory.plot_structural_virality('test_dataset', '1160842257647493120')
        fig.show()

    def test_plot_propagation_tree(self):
        fig = self.plot_factory.plot_propagation_tree('test_dataset', '1160842257647493120')
        fig.show()

    def test_graph_propagation(self):
        graph = self.plot_factory.get_full_graph('test_dataset')
        components = graph.connected_components(mode='weak')
        self.assertEqual(len(components), 8485)

    def test_depth_cascade_ccdf_plot(self):
        fig = self.plot_factory.plot_depth_cascade_ccdf('test_dataset')
        fig.show()

    def test_size_cascade_ccdf_plot(self):
        fig = self.plot_factory.plot_size_cascade_ccdf('test_dataset')
        fig.show()

    def test_cascade_count_over_time_plot(self):
        fig = self.plot_factory.plot_cascade_count_over_time('test_dataset')
        fig.show()


if __name__ == '__main__':
    unittest.main()
