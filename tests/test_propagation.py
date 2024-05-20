import unittest

import igraph as ig
import matplotlib.pyplot as plt
from pandas import Timestamp
from pymongo import MongoClient

from figures.propagation import PropagationPlotFactory
from tests.conftest import populate_test_database, delete_test_database
import igraph as ig


class PropagationTestCase(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     populate_test_database('test_dataset')

    #
    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database('test_dataset')

    def setUp(self):
        self.plot_factory = PropagationPlotFactory(available_datasets=['test_dataset'])

    def test_propagation_tree(self):
        graph = self.plot_factory.get_propagation_tree('test_dataset', '1160842257647493120')
        self.assertEqual(graph.vcount(), 76)
        self.assertEqual(graph.ecount(), 61)
        self.assertEqual(graph.is_directed(), True)
        self.assertIsInstance(graph.vs['author_id'][0], str)
        self.assertIsInstance(graph.vs['username'][0], str)
        self.assertIsInstance(graph.vs['tweet_id'][0], str)
        self.assertEqual(graph.vs['party'][0], None)
        self.assertEqual(graph.vs['is_usual_suspect'][0], False)
        self.assertEqual(graph.vs['created_at'][0], Timestamp('2019-08-10 10:09:05+0000', tz='UTC'))

        # Display the igraph graph with matplotlib
        layout = graph.layout(self.plot_factory.layout)
        fig, ax = plt.subplots()
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

    def test_propagation_lengths(self):
        df = self.plot_factory.get_conversation_lengths('test_dataset')
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
        collection.insert_many([{
            'id': i,
            'conversation_id': 0,
            'referenced_tweets': [{'type': 'replied_to', 'id': j}]
        } for i, j in edges])
        graph = self.plot_factory.get_propagation_tree(dataset, 0)
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set((graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es)
        self.assertEqual(actual_edges, set(edges))

    def test_propagation_tree_plot(self):
        graph = self.plot_factory.get_propagation_tree('test_dataset', '1160842257647493120')
        fig = self.plot_factory.plot_network(graph)
        fig.show()

    def test_depth_plot(self):
        fig = self.plot_factory.plot_depth('test_dataset', '1160842257647493120')
        fig.show()

    def test_size_plot(self):
        fig = self.plot_factory.plot_size('test_dataset', '1160842257647493120')
        fig.show()

    def test_plot_propagation_tree(self):
        fig = self.plot_factory.plot_propagation_tree('test_dataset', '1160842257647493120')
        fig.show()


if __name__ == '__main__':
    unittest.main()
