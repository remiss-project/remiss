import unittest

import igraph as ig
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timestamp
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all

from propagation import DiffusionMetrics, Egonet

patch_all()


class DiffusionMetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.egonet = Egonet()
        self.diffusion_metrics = DiffusionMetrics(egonet=self.egonet)
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'  # str(uuid.uuid4().hex)
        self.test_tweet_id = '1167074391315890176'
        self.missing_tweet_id = '1077146799692021761'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        if self.tmp_dataset in client.list_database_names():
            client.drop_database(self.tmp_dataset)

    def test_get_references_simple(self):
        client = MongoClient('localhost', 27017)

        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        timestamps = [Timestamp.now() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    } for i in range(8)]
        data = [{
            'id': str(j),
            'author': authors[j],
            'text': f'Tweet {j}',
            'conversation_id': '0',
            'referenced_tweets': [{'type': 'retweeted',
                                   'id': str(i),
                                   'author': authors[i],
                                   'created_at': timestamps[i],
                                   'text': f'Tweet {i}'
                                   }],
            'created_at': timestamps[j]

        } for i, j in edges]
        collection.insert_many(data)
        resources = self.diffusion_metrics.get_references(self.tmp_dataset, '0')
        assert not pd.isna(resources).any().any()
        self.assertEqual(resources[['source', 'target']].astype(int).to_records(index=False).tolist(), edges)

    def test_get_vertices_from_edges_simple(self):
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
        timestamps = [Timestamp.today().date() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    } for i in range(8)]
        types = {0: 'original', 1: 'retweeted', 2: 'quoted', 3: 'replied_to', 4: 'retweeted', 5: 'quoted',
                 6: 'replied_to', 7: 'retweeted'}
        data = [{
            'id': str(j),
            'author': authors[j],
            'text': f'Tweet {j}',
            'conversation_id': '0',
            'referenced_tweets': [{'type': types[j],
                                   'id': str(i),
                                   'author': authors[i],
                                   'created_at': timestamps[i],
                                   'text': f'Tweet {i}'
                                   }],
            'created_at': timestamps[j]

        } for i, j in edges]
        collection.insert_many(data)
        edges = self.diffusion_metrics.get_references(self.tmp_dataset, '0')
        vertices = self.diffusion_metrics._get_vertices_from_edges(edges, '0')
        assert not pd.isna(vertices).any().any()
        expected = {'author_id': {'0': 'author_id_0', '1': 'author_id_1', '2': 'author_id_2', '3': 'author_id_3',
                                  '4': 'author_id_4', '5': 'author_id_5', '6': 'author_id_6', '7': 'author_id_7'},
                    'created_at': {'0': Timestamp('2024-10-11 00:00:00+0000', tz='UTC'),
                                   '1': Timestamp('2024-10-11 01:00:00+0000', tz='UTC'),
                                   '2': Timestamp('2024-10-11 02:00:00+0000', tz='UTC'),
                                   '3': Timestamp('2024-10-11 03:00:00+0000', tz='UTC'),
                                   '4': Timestamp('2024-10-11 04:00:00+0000', tz='UTC'),
                                   '5': Timestamp('2024-10-11 05:00:00+0000', tz='UTC'),
                                   '6': Timestamp('2024-10-11 06:00:00+0000', tz='UTC'),
                                   '7': Timestamp('2024-10-11 07:00:00+0000', tz='UTC')},
                    'text': {'0': 'Tweet 0', '1': 'Tweet 1', '2': 'Tweet 2', '3': 'Tweet 3', '4': 'Tweet 4',
                             '5': 'Tweet 5', '6': 'Tweet 6', '7': 'Tweet 7'},
                    'type': {'0': 'original', '1': 'retweeted', '2': 'quoted', '3': 'replied_to', '4': 'retweeted',
                             '5': 'quoted', '6': 'replied_to', '7': 'retweeted'},
                    'username': {'0': 'username_0', '1': 'username_1', '2': 'username_2', '3': 'username_3',
                                 '4': 'username_4', '5': 'username_5', '6': 'username_6', '7': 'username_7'}}
        self.assertEqual(vertices.to_dict(), expected)

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
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    } for i in range(8)]
        data = [{
            'id': str(j),
            'author': authors[j],
            'text': f'Tweet {j}',
            'conversation_id': '0',
            'referenced_tweets': [{'type': 'retweeted',
                                   'id': str(i),
                                   'author': authors[i],
                                   'created_at': timestamps[i],
                                   'text': f'Tweet {i}'
                                   }],
            'created_at': timestamps[j]

        } for i, j in edges]
        collection.insert_many(data)
        graph = self.diffusion_metrics.compute_propagation_tree(self.tmp_dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax, vertex_label=graph.vs['tweet_id'])
        plt.show()
        actual_edges = set(
            (graph.vs['author_id'][edge.source], graph.vs['author_id'][edge.target]) for edge in graph.es)
        expected_edges = {(f'author_id_{source}', f'author_id_{target}') for source, target in edges}
        self.assertEqual(actual_edges, expected_edges)

    def test_references_local(self):
        references = self.diffusion_metrics.get_references(self.test_dataset, self.test_tweet_id)

        expected = ['source',
                    'target',
                    'source_author_id',
                    'source_username',
                    'target_author_id',
                    'target_username',
                    'source_text',
                    'target_text',
                    'type',
                    'source_created_at',
                    'target_created_at']
        self.assertEqual(references.columns.to_list(), expected)
        sources = references['source'].unique()
        self.assertGreater(len(sources), 1)
        self.assertEqual(references['type'].unique().tolist(), ['retweeted', 'quoted', 'replied_to'])

    def test_get_vertices_from_edges_local(self):
        references = self.diffusion_metrics.get_references(self.test_dataset, self.test_tweet_id)

        vertices = self.diffusion_metrics._get_vertices_from_edges(references, self.test_tweet_id)
        expected = ['author_id', 'username', 'text', 'created_at', 'type']
        self.assertEqual(vertices.columns.to_list(), expected)
        self.assertIn('original', vertices['type'].unique())

    def test_propagation_tree_local(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

    @unittest.skip('Remote version')
    def test_references_remote(self):
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        references = self.diffusion_metrics.get_references('Andalucia_2022', test_tweet_id)

        expected = ['source',
                    'target',
                    'source_author_id',
                    'source_username',
                    'target_author_id',
                    'target_username',
                    'source_text',
                    'target_text',
                    'type',
                    'created_at']
        self.assertEqual(references.columns.to_list(), expected)
        sources = references['source'].unique()
        self.assertGreater(len(sources), 1)
        self.assertEqual(references['type'].unique().tolist(), ['retweeted', 'quoted', 'replied_to'])

        self.diffusion_metrics.host = 'localhost'

    @unittest.skip('Remote version')
    def test_get_vertices_from_edges_remote(self):
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        references = self.diffusion_metrics.get_references('Andalucia_2022', test_tweet_id)

        vertices = self.diffusion_metrics._get_vertices_from_edges(references)
        expected = ['tweet_id', 'author_id', 'username', 'text', 'created_at', 'type']
        self.assertEqual(vertices.columns.to_list(), expected)

        self.diffusion_metrics.host = 'localhost'

    @unittest.skip('Remote version')
    def test_propagation_tree_remote(self):
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        self.diffusion_metrics.egonet.host = 'mongodb://srvinv02.esade.es'
        graph = self.diffusion_metrics.compute_propagation_tree('Andalucia_2022', test_tweet_id)
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

        self.diffusion_metrics.host = 'localhost'

    @unittest.skip('Remote version')
    def test_propagation_tree_remote_2(self):
        test_tweet_id = '1182192005377601536'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        self.diffusion_metrics.egonet.host = 'mongodb://srvinv02.esade.es'

        graph = self.diffusion_metrics.compute_propagation_tree('Openarms', test_tweet_id)
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

        self.diffusion_metrics.host = 'localhost'

    def test_depth(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        # plot graph
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax, vertex_label=graph.vs['tweet_id'])
        plt.show()
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
        df = self.diffusion_metrics.compute_depth_over_time(shortest_paths)
        self.assertEqual(df.shape[0], 92)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 4)

    def test_size(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        df = self.diffusion_metrics.compute_size_over_time(graph)
        self.assertEqual(df.shape[0], 92)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 92)

    def test_max_breadth(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
        df = self.diffusion_metrics.compute_max_breadth_over_time(shortest_paths)
        self.assertEqual(df.shape[0], 92)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 84)

    def test_structured_virality(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        df = self.diffusion_metrics.compute_structural_virality_over_time(graph)
        self.assertEqual(df.shape[0], 92)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertAlmostEqual(df.max(), 4.442143326758711)

    def test_precomputed_depth(self):
        df = self.diffusion_metrics.get_depth_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 91)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 7)

    def test_precomputed_size(self):
        df = self.diffusion_metrics.get_size_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 92)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 92)

    def test_precomputed_max_breadth(self):
        df = self.diffusion_metrics.get_max_breadth_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 91)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 28)

    def test_precomputed_structured_virality(self):
        df = self.diffusion_metrics.get_structural_virality_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 92)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 4.442143326758711)

    def test_size_cascade_ccdf(self):
        start_time = Timestamp.now()
        df = self.diffusion_metrics.get_size_cascade_ccdf(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        self.assertEqual(df.columns.to_list(), ['Normal', 'Politician'])

    def test_get_cascade_sizes(self):
        start_time = Timestamp.now()
        cascade_sizes = self.diffusion_metrics.get_cascade_sizes(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        self.assertEqual(cascade_sizes.shape[0], 526)
        self.assertEqual(['is_usual_suspect', 'party', 'size'], cascade_sizes.columns.to_list())

    def test_cascade_count_over_time(self):
        start_time = Timestamp.now()
        df = self.diffusion_metrics.get_cascade_count_over_time(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        self.assertEqual('Cascade Count', df.name)

    def test_persistence(self):
        print(f'Persisting on {self.tmp_dataset}')
        client = MongoClient('localhost', 27017)

        client.drop_database(self.tmp_dataset)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        num_conversations = 4
        num_vertices = 6
        test_data = []
        cascade_ids = []
        for cascade_id in range(num_conversations):
            tree = ig.Graph.Tree_Game(num_vertices, directed=True)
            edges = tree.get_edgelist()

            timestamps = [Timestamp.today().date() + pd.offsets.Hour(i) for i in range(num_vertices)]
            authors = [{'id': f'author_id_{i}',
                        'username': f'username_{i}',
                        } for i in range(num_vertices)]
            types = {0: 'original', 1: 'retweeted', 2: 'quoted', 3: 'replied_to', 4: 'retweeted', 5: 'quoted',
                     6: 'replied_to', 7: 'retweeted'}
            data = []
            # add original tweet
            original_tweet = {
                'id': str(cascade_id * num_vertices),
                'author': authors[0],
                'text': 'Original tweet',
                'conversation_id': '0',
                'created_at': timestamps[0]
            }
            cascade_ids.append(str(cascade_id * num_vertices))
            data.append(original_tweet)
            for i, j in edges:
                tweet = {
                    'id': str(j + cascade_id * num_vertices),
                    'author': authors[j],
                    'text': f'Tweet {j}',
                    'conversation_id': '0',
                    'created_at': timestamps[j]
                }
                if types[j] != 'original':
                    tweet['referenced_tweets'] = [{'type': types[j],
                                                   'id': str(i),
                                                   'author': authors[i],
                                                   'created_at': timestamps[i],
                                                   'text': f'Tweet {i}'
                                                   }]

                data.append(tweet)
            test_data.append(data)
            collection.insert_many(data)
        client.close()

        # Test
        self.diffusion_metrics.n_jobs = 1
        self.diffusion_metrics.persist([self.tmp_dataset])

        for cascade_id in cascade_ids:
            expected_tree = self.diffusion_metrics.compute_propagation_tree(self.tmp_dataset, cascade_id)
            actual_tree = self.diffusion_metrics.get_propagation_tree(self.tmp_dataset, cascade_id)

            self.assertEqual(expected_tree.vcount(), actual_tree.vcount())
            self.assertEqual(expected_tree.ecount(), actual_tree.ecount())
            self.assertTrue(expected_tree.isomorphic(actual_tree))

            expected_attributes = pd.DataFrame(
                {attribute: expected_tree.vs[attribute] for attribute in expected_tree.vs.attributes()})
            actual_attributes = pd.DataFrame(
                {attribute: actual_tree.vs[attribute] for attribute in actual_tree.vs.attributes()})
            expected_attributes = expected_attributes.set_index('tweet_id').sort_index()
            actual_attributes = actual_attributes.set_index('tweet_id').sort_index()
            pd.testing.assert_frame_equal(expected_attributes, actual_attributes, check_dtype=False,
                                          check_index_type=False, check_column_type=False)
            expected_shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(
                expected_tree)
            expected_depth = self.diffusion_metrics.compute_depth_over_time(expected_shortest_paths)
            actual_depth = self.diffusion_metrics.get_depth_over_time(self.tmp_dataset, cascade_id)

            self.assertEqual(expected_depth.shape, actual_depth.shape)
            self.assertTrue(expected_depth.equals(actual_depth))

            expected_size = self.diffusion_metrics.get_size_over_time(self.tmp_dataset, cascade_id)
            actual_size = self.diffusion_metrics.compute_size_over_time(expected_tree)

            self.assertEqual(expected_size.shape, actual_size.shape)
            self.assertTrue(expected_size.equals(actual_size))

            expected_max_breadth = self.diffusion_metrics.get_max_breadth_over_time(self.tmp_dataset, cascade_id)
            actual_max_breadth = self.diffusion_metrics.compute_max_breadth_over_time(expected_shortest_paths)

            self.assertEqual(expected_max_breadth.shape, actual_max_breadth.shape)
            self.assertTrue(expected_max_breadth.equals(actual_max_breadth))

            expected_structural_virality = self.diffusion_metrics.get_structural_virality_over_time(
                self.tmp_dataset,
                cascade_id)
            actual_structural_virality = self.diffusion_metrics.compute_structural_virality_over_time(expected_tree)

            self.assertEqual(expected_structural_virality.shape, actual_structural_virality.shape)
            pd.testing.assert_series_equal(expected_structural_virality, actual_structural_virality,
                                           check_dtype=False, atol=1e-6, rtol=1e-6)

    def test_persistence_size_cascade_ccdf(self):
        start_time = Timestamp.now()
        self.diffusion_metrics._persist_plot_size_cascade_ccdf_to_mongodb(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        actual = self.diffusion_metrics._load_plot_size_cascade_ccdf_from_mongodb(self.test_dataset)
        expected = self.diffusion_metrics.get_size_cascade_ccdf(self.test_dataset)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, atol=1e-6, rtol=1e-6, check_index_type=False)

    def test_persistence_cascade_count_over_time(self):
        start_time = Timestamp.now()
        self.diffusion_metrics._persist_plot_cascade_count_over_time_to_mongodb(self.test_dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        actual = self.diffusion_metrics._load_plot_cascade_count_over_time_from_mongodb(self.test_dataset)
        expected = self.diffusion_metrics.get_cascade_count_over_time(self.test_dataset)
        self.assertTrue((expected.index == actual.index).all())
        self.assertTrue((expected == actual).all())

    @unittest.skip('Remote version')
    def test_get_shortest_path_remote(self):
        test_tweet = '1182192005377601536'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        self.diffusion_metrics.egonet.host = 'mongodb://srvinv02.esade.es'
        graph = self.diffusion_metrics.compute_propagation_tree('Openarms', test_tweet)
        fig, ax = plt.subplots(figsize=(20, 20))
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax, node_size=3, vertex_label=graph.vs['tweet_id'], arrow_size=10)
        plt.show()
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
        self.assertFalse(shortest_paths.isna().any().any())

    def test_get_shortest_path_2(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)

        self.assertIn(self.test_tweet_id, [vertex['tweet_id'] for vertex in graph.vs])
        # fig, ax = plt.subplots(figsize=(20, 20))
        # layout = graph.layout('fr')
        # color_type = {'original': 'blue',
        #               'retweeted': 'green',
        #               'quoted': 'orange',
        #               'replied_to': 'red',
        #               'other': 'black'}
        # color = [color_type[vertex['type']] for vertex in graph.vs]
        # ig.plot(graph, layout=layout, target=ax, node_size=3, vertex_label=graph.vs['username'], arrow_size=10,
        #         vertex_color=color)
        # plt.show()
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
        self.assertFalse(shortest_paths.iloc[1:].isna().any().any())

    def test_depth_3(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)

        fig, ax = plt.subplots(figsize=(20, 20))
        layout = graph.layout('fr')
        color_type = {'original': 'blue',
                      'retweeted': 'green',
                      'quoted': 'orange',
                      'replied_to': 'red',
                      'other': 'black'}
        color = [color_type[vertex['type']] for vertex in graph.vs]
        ig.plot(graph, layout=layout, target=ax, node_size=3, vertex_label=graph.vs['username'], arrow_size=10,
                vertex_color=color)
        plt.show()
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
        depth_over_time = self.diffusion_metrics.compute_depth_over_time(shortest_paths)
        self.assertFalse(depth_over_time.isna().any().any())

    def test_no_nats_persistence(self):
        test_tweet_id = '1165265987370983424'
        expected = self.diffusion_metrics._compute_cascade_metrics_for_persistence(self.test_dataset, test_tweet_id)
        client = MongoClient('localhost', 27017)
        database = client.get_database('test_dataset_2')
        collection = database.get_collection('diffusion_metrics')
        collection.insert_many([expected])
        actual = self.diffusion_metrics.load_cascade_data(self.test_dataset, test_tweet_id)
        del expected['_id']
        del actual['_id']
        actual['vs_attributes']['created_at'] = [pd.Timestamp(actual['vs_attributes']['created_at'][0])]

        self.assertEqual(expected, actual)

    def test_depth_2(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, '1167100545800318976')
        # plot graph
        self.diffusion_metrics._plot_graph_igraph(graph)
        shortest_paths = self.diffusion_metrics.get_shortest_paths_to_original_tweet_over_time(graph)
        df = self.diffusion_metrics.compute_depth_over_time(shortest_paths)

        self.assertEqual(df.max(), 4)


if __name__ == '__main__':
    unittest.main()
