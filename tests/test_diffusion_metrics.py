import unittest
import uuid

import igraph as ig
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timestamp
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all

from propagation import DiffusionMetrics

patch_all()


class DiffusionMetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.diffusion_metrics = DiffusionMetrics()
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'# str(uuid.uuid4().hex)
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
        vertices = self.diffusion_metrics._get_vertices_from_edges(edges)
        assert not pd.isna(vertices).any().any()
        expected = {'author_id': {0: 'author_id_0', 1: 'author_id_1', 2: 'author_id_2', 3: 'author_id_3',
                                  4: 'author_id_4', 5: 'author_id_5', 6: 'author_id_6', 7: 'author_id_7'},
                    'created_at': {i: timestamps[i] for i in range(8)},
                    'text': {0: 'Tweet 0', 1: 'Tweet 1', 2: 'Tweet 2', 3: 'Tweet 3', 4: 'Tweet 4', 5: 'Tweet 5',
                             6: 'Tweet 6', 7: 'Tweet 7'},
                    'tweet_id': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7'},
                    'type': {0: 'original', 1: 'retweeted', 2: 'quoted', 3: 'replied_to', 4: 'retweeted', 5: 'quoted',
                             6: 'replied_to', 7: 'retweeted'},
                    'username': {0: 'username_0', 1: 'username_1', 2: 'username_2', 3: 'username_3', 4: 'username_4',
                                 5: 'username_5', 6: 'username_6', 7: 'username_7'}}
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

    def test_get_vertices_from_edges_remote(self):
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        references = self.diffusion_metrics.get_references('Andalucia_2022', test_tweet_id)

        vertices = self.diffusion_metrics._get_vertices_from_edges(references)
        expected = ['tweet_id', 'author_id', 'username', 'text']
        self.assertEqual(vertices.columns.to_list(), expected)

        self.diffusion_metrics.host = 'localhost'

    def test_propagation_tree_remote(self):
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        graph = self.diffusion_metrics.compute_propagation_tree('Andalucia_2022', test_tweet_id)
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

        self.diffusion_metrics.host = 'localhost'

    # def test_fix_wrong_timestamps(self):
    #     client = MongoClient('localhost', 27017)
    #     raw = client.get_database(self.test_dataset).get_collection('raw')
    #
    #     # go through all tweets and fix the timestamp formate to datetime of the referenced tweets
    #     all_tweets = raw.find()
    #     for tweet in all_tweets:
    #         if 'referenced_tweets' in tweet:
    #             changed = False
    #             for referenced_tweet in tweet['referenced_tweets']:
    #                 if 'created_at' in referenced_tweet:
    #                     referenced_tweet['created_at'] = Timestamp(referenced_tweet['created_at'])
    #                     changed = True
    #             if changed:
    #                 raw.replace_one({'id': tweet['id']}, tweet)

    def test_depth(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        # plot graph
        fig, ax = plt.subplots()
        layout = graph.layout('fr')
        ig.plot(graph, layout=layout, target=ax, vertex_label=graph.vs['tweet_id'])
        plt.show()
        df = self.diffusion_metrics.compute_depth_over_time(graph)
        self.assertEqual(df.shape[0], 94)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 1)

    def test_size(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        df = self.diffusion_metrics.compute_size_over_time(graph)
        self.assertEqual(df.shape[0], 94)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 94)

    def test_max_breadth(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        df = self.diffusion_metrics.compute_max_breadth_over_time(graph)
        self.assertEqual(df.shape[0], 94)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 94)

    def test_structured_virality(self):
        graph = self.diffusion_metrics.compute_propagation_tree(self.test_dataset, self.test_tweet_id)
        df = self.diffusion_metrics.compute_structural_virality_over_time(graph)
        self.assertEqual(df.shape[0], 94)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertAlmostEqual(df.max(), 1.9782707107288358)

    def test_precomputed_depth(self):
        df = self.diffusion_metrics.get_depth_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 12)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 1)

    def test_precomputed_size(self):
        df = self.diffusion_metrics.get_size_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 12)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 12)

    def test_precomputed_max_breadth(self):
        df = self.diffusion_metrics.get_max_breadth_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 12)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 11)

    def test_precomputed_structured_virality(self):
        df = self.diffusion_metrics.get_structural_virality_over_time(self.test_dataset, self.test_tweet_id)
        self.assertEqual(df.shape[0], 12)
        self.assertIsInstance(df.index, pd.DatetimeIndex)
        self.assertEqual(df.max(), 1.6805555555555554)

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
        self.assertEqual(cascade_sizes.shape[0], 220)
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
        num_conversations = 10
        num_vertices = 8
        num_edges = 10
        test_data = []
        cascade_ids = []
        for cascade_id in range(num_conversations):
            edges = np.random.randint(0, num_vertices, (num_edges, 2))
            edges = edges[edges[:, 0] != edges[:, 1]]

            timestamps = [Timestamp.today().date() + pd.offsets.Hour(i) for i in range(num_vertices)]
            authors = [{'id': f'author_id_{i}',
                        'username': f'username_{i}',
                        } for i in range(num_vertices)]
            types = {0: 'original', 1: 'retweeted', 2: 'quoted', 3: 'replied_to', 4: 'retweeted', 5: 'quoted',
                     6: 'replied_to', 7: 'retweeted'}
            data = []
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
                else:
                    cascade_ids.append(tweet['id'])
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
            self.assertTrue(expected_attributes.equals(actual_attributes))

            expected_depth = self.diffusion_metrics.compute_depth_over_time(expected_tree)
            actual_depth = self.diffusion_metrics.get_depth_over_time(self.tmp_dataset, cascade_id)

            self.assertEqual(expected_depth.shape, actual_depth.shape)
            self.assertTrue(expected_depth.equals(actual_depth))

            expected_size = self.diffusion_metrics.get_size_over_time(self.tmp_dataset, cascade_id)
            actual_size = self.diffusion_metrics.compute_size_over_time(expected_tree)

            self.assertEqual(expected_size.shape, actual_size.shape)
            self.assertTrue(expected_size.equals(actual_size))

            expected_max_breadth = self.diffusion_metrics.get_max_breadth_over_time(self.tmp_dataset, cascade_id)
            actual_max_breadth = self.diffusion_metrics.compute_max_breadth_over_time(expected_tree)

            self.assertEqual(expected_max_breadth.shape, actual_max_breadth.shape)
            self.assertTrue(expected_max_breadth.equals(actual_max_breadth))

            expected_structural_virality = self.diffusion_metrics.get_structural_virality_over_time(
                self.tmp_dataset,
                cascade_id)
            actual_structural_virality = self.diffusion_metrics.compute_structural_virality_over_time(expected_tree)

            self.assertEqual(expected_structural_virality.shape, actual_structural_virality.shape)
            pd.testing.assert_series_equal(expected_structural_virality, actual_structural_virality,
                                           check_dtype=False, atol=1e-6, rtol=1e-6)

    def test_propagation_data_original(self):
        client = MongoClient('mongodb://srvinv02.esade.es', 27017)
        raw = client.get_database('Andalucia_2022').get_collection('raw')

        # all_tweets_by_text_pipeline = [
        #     {'$unwind': '$referenced_tweets'},
        #     {'$group': {'_id': '$text', 'targets': {'$push': '$id'}, 'count': {'$sum': 1},
        #                 'sources': {'$push': '$referenced_tweets.id'}}},
        #     # {'$match': {'_id': {'$regex': text}}},
        #     {'$sort': {'count': -1}},
        #     {'$project': {'_id': 0, 'targets': 1, 'text': '$_id', 'count': 1, 'sources': 1}},
        # ]
        # all_tweets = raw.aggregate_pandas_all(all_tweets_by_text_pipeline)
        # all_tweets['sources'] = all_tweets['sources'].apply(lambda x: set(x))
        #
        # sources = {'1111662918016356352', '1111663197445152770', '1111663499795681280', '1111663349702578177',
        #            '1111663971336101889', '1111662671210926086', '1111663702003146754', '1111664600955658240',
        #            '1111662296001142785'}

        # find original among sources
        original_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$project': {'id': 1, 'text': 1, 'created_at': 1, 'author_id': 1}}
        ]

        original = raw.aggregate_pandas_all(original_pipeline)

        test_tweet_id = '1076590984941764608'
        test_tweet_id = '975695555375583232'
        original, references = self.diffusion_metrics.get_references(self.test_dataset, test_tweet_id)

        self.assertEqual(original.index.to_list(), ['tweet_id', 'author_id', 'created_at'])
        self.assertEqual(original['tweet_id'], test_tweet_id)
        self.assertEqual(references.columns.to_list(),
                         ['source', 'target', 'text', 'type', 'source_author', 'target_author', 'created_at'])

    def test_propagation_data_retweet(self):
        test_tweet_id = '1076621980940623873'
        references = self.diffusion_metrics.get_references(self.test_dataset, test_tweet_id)

        self.assertEqual(references.columns.to_list(),
                         ['source', 'target', 'text', 'type', 'source_author', 'target_author', 'created_at'])

    def test_propagation_data_missing(self):
        test_tweet_id = '1014595663106060289'
        references = self.diffusion_metrics.get_references(self.test_dataset, test_tweet_id)

        self.assertEqual(references.columns.to_list(),
                         ['source', 'target', 'text', 'type', 'source_author', 'target_author', 'created_at'])

    def test_propagation_data_original_remote(self):
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        references = self.diffusion_metrics.get_references('Andalucia_2022', test_tweet_id)

        self.assertEqual(references.columns.to_list(),
                         ['source',
                          'target',
                          'source_author_id',
                          'source_username',
                          'target_author_id',
                          'target_username',
                          'text',
                          'type',
                          'created_at'])

        self.diffusion_metrics.host = 'localhost'

    def test_propagation_data_missing_remote(self):
        test_tweet_id = '1564253092832546816'
        test_tweet_id = '1523011526445211649'
        self.diffusion_metrics.host = 'mongodb://srvinv02.esade.es'
        references = self.diffusion_metrics.get_references('Andalucia_2022', test_tweet_id)

        self.assertEqual(references.columns.to_list(),
                         ['source', 'target', 'text', 'type', 'source_author', 'target_author', 'created_at'])

        self.diffusion_metrics.host = 'localhost'

    def test_propagate_propagation(self):
        prop = Propagation(tweet_id='1523011526445211649', dataset='Andalucia_2022',
                           host='mongodb://srvinv02.esade.es')
        prop.propagate()
        print(prop.edges)

    def test_propagate_propagation_local(self):
        prop = Propagation(tweet_id='1167137262515163136', dataset=self.test_dataset,
                           )
        prop.propagate()
        print(prop.edges)

    def test_missing_rts_local(self):
        test_tweet_id = self.test_tweet_id
        rt = 'RT @matteosalvinimi: '
        text = 'Questa nave @openarms_fund si trova in acqu'
        reference_types = ['retweeted', 'quoted', 'replied_to']

        client = MongoClient('localhost', 27017)
        raw = client.get_database(self.test_dataset).get_collection('raw')

        all_tweets_by_text_pipeline = [
            {'$group': {'_id': '$text', 'ids': {'$push': '$id'}, 'count': {'$sum': 1},
                        'rt_ids': {'$push': '$referenced_tweets.id'}}},
            # {'$match': {'_id': {'$regex': text}}},
            {'$sort': {'count': -1}},
            {'$project': {'_id': 0, 'ids': 1, 'text': '$_id', 'count': 1, 'rt_ids': 1}},
        ]
        all_tweets = raw.aggregate_pandas_all(all_tweets_by_text_pipeline)

        # match all tweets containing the text of the tweet
        pipeline = [
            {'$match': {'text': {'$regex': text}}},

            {'$project': {'_id': 0, 'id': 1, 'text': 1, 'truncated': 1}},
        ]
        original = raw.aggregate_pandas_all(pipeline).iloc[0]
        original_tweet = raw.find_one({'id': original['id']})

        original_tweets = raw.aggregate_pandas_all([
            {'$match': {'referenced_tweets': {'$exists': False}}},
        ])

        pipeline = [
            {'$match': {'text': {'$regex': 'RT.*' + text}}},
            {'$project': {'_id': 0, 'id': 1, 'text': 1, 'public_metrics': 1, 'truncated': 1}},
        ]

        rts = raw.aggregate_pandas_all(pipeline)
        print(original['id'], original['text'])
        print(rts['id'], rts['text'])
        print(len(rts), original['public_metrics']['retweet_count'])


if __name__ == '__main__':
    unittest.main()
