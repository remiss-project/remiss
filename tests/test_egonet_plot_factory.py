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

from figures.propagation import PropagationPlotFactory, compute_backbone


class TestEgonetPlotFactory(unittest.TestCase):
    def setUp(self):
        self.plot_factory = PropagationPlotFactory()
        self.host = 'localhost'
        self.port = 27017
        self.plot_factory.host = self.host
        self.plot_factory.port = self.port
        self.dataset = 'test_dataset'

    def tearDown(self) -> None:
        client = MongoClient('localhost', 27017)
        client.drop_database(self.dataset)
        if self.plot_factory.cache_dir:
            shutil.rmtree(self.plot_factory.cache_dir)

    def test_plot_egonet(self):
        # Mock get_egonet
        network = ig.Graph.GRG(8, 0.2)
        network.vs['author_id'] = ['0', '1', '2', '3', '4', '5', '6', '7']
        network.vs['username'] = ['TEST_USER_0', 'TEST_USER_1', 'TEST_USER_2', 'TEST_USER_3', 'TEST_USER_4',
                                  'TEST_USER_5', 'TEST_USER_6', 'TEST_USER_7']
        network.vs['party'] = ['PSOE', None, 'VOX', None, 'PSOE', None, 'VOX', None]
        network.vs['is_usual_suspect'] = [False, False, False, False, True, True, True, True]
        network['reputation'] = pd.DataFrame({'author_id': ['0', '1', '2', '3', '4', '5', '6', '7'],
                                              datetime(2020, 1, 1): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                              }).set_index('author_id')

        network.vs['legitimacy'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.plot_factory.get_egonet = Mock(return_value=network)

        collection = 'test_collection'

        actual = self.plot_factory.plot_egonet(collection, 'test_user', 1)
        self.assertEqual(len(actual['data'][0]['x']), network.ecount() * 3)
        self.assertEqual(len(actual['data'][0]['y']), network.ecount() * 3)
        self.assertEqual(len(actual['data'][0]['z']), network.ecount() * 3)
        self.assertEqual(len(actual['data'][1]['x']), network.vcount())
        self.assertEqual(len(actual['data'][1]['y']), network.vcount())
        self.assertEqual(len(actual['data'][1]['z']), network.vcount())
    @patch('figures.propagation.MongoClient')
    def test_check_color_coding(self, mock_mongo_client):

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
            elif 'legitimacy' in pipeline[-1]['$project']:
                legitimacy = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                           'date': [datetime(2020, 1, 1),
                                                    datetime(2020, 1, 2),
                                                    datetime(2020, 1, 3),
                                                    datetime(2020, 1, 4)],
                                           'legitimacy': [0.1, 0.2, 0.3, 0.4]})
                return legitimacy
            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                        'is_usual_suspect': [False, False, True, True],
                                        'party': [None, 'PSOE', None, 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683'],
                                        'num_tweets': [1, 2, 3, 4],
                                        'num_followers': [100, 200, 300, 400],
                                        'num_following': [10, 20, 30, 40],
                                        })
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected_colors = [0.1, 0.2, 0.3, 0.4]

        plot = self.plot_factory.plot_egonet(collection, None, 1, start_date=None, end_date=None)
        self.assertEqual(list(plot['data'][1]['marker']['color']), expected_colors)

    @patch('figures.propagation.MongoClient')
    def test_check_size_coding(self, mock_mongo_client):

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
            elif 'date' in pipeline[-1]['$project']:
                legitimacy = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                           'date': [datetime(2020, 1, 1),
                                                    datetime(2020, 1, 2),
                                                    datetime(2020, 1, 3),
                                                    datetime(2020, 1, 4)],
                                           'legitimacy': [0.1, 0.2, 0.3, 0.4]})
                return legitimacy
            elif 'legitimacy' in pipeline[-1]['$project']:
                legitimacy = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                           'legitimacy': [0.2, 0.3, 0.4, 0.5]})
                return legitimacy
            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                        'is_usual_suspect': [False, False, True, True],
                                        'party': [None, None, None, 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683'],
                                        'num_tweets': [1, 2, 3, 4],
                                        'num_followers': [100, 200, 300, 400],
                                        'num_following': [10, 20, 30, 40],
                                        })
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected_colors = [0.2, 0.3, 0.4, 0.5]
        expected_sizes = [9.56521739130435, 10.0, 10.0, 9.56521739130435]

        plot = self.plot_factory.plot_egonet(collection, None, 1)
        self.assertEqual(list(plot['data'][1]['marker']['color']), expected_colors)
        self.assertEqual(list(plot['data'][1]['marker']['size']), expected_sizes)

    @patch('figures.propagation.MongoClient')
    def test_check_marker_coding(self, mock_mongo_client):
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
            elif 'date' in pipeline[-1]['$project']:
                legitimacy = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                           'date': [datetime(2020, 1, 1),
                                                    datetime(2020, 1, 2),
                                                    datetime(2020, 1, 3),
                                                    datetime(2020, 1, 4)],
                                           'legitimacy': [0.1, 0.2, 0.3, 0.4]})
                return legitimacy
            elif 'legitimacy' in pipeline[-1]['$project']:
                legitimacy = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                           'legitimacy': [0.2, 0.3, 0.4, 0.5]})
                return legitimacy
            else:
                # its authors
                authors = pd.DataFrame({'author_id': ['1', '2', '3', '4'],
                                        'is_usual_suspect': [False, True, False, True],
                                        'party': [None, None, 'PP', 'VOX'],
                                        'username': ['TEST_USER_488680', 'TEST_USER_488681', 'TEST_USER_488682',
                                                     'TEST_USER_488683'],
                                        'num_tweets': [1, 2, 3, 4],
                                        'num_followers': [100, 200, 300, 400],
                                        'num_following': [10, 20, 30, 40],
                                        })
                return authors

        mock_collection.aggregate_pandas_all = aggregate_pandas_all
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        collection = 'test_collection'

        expected = ['circle', 'diamond', 'square', 'cross']

        plot = self.plot_factory.plot_egonet(collection, None, 1)
        self.assertEqual(list(plot['data'][1]['marker']['symbol']), expected)

    def test_no_isolated_vertices(self):
        # Checks it returns the whole thing if the user is not present
        self.plot_factory.simplification = 'backbone'
        self.plot_factory.threshold = 0.232
        self.plot_factory.cache_dir = None
        data_size = 1000
        max_num_references = 10
        user = 'test_user'
        depth = 1
        test_data = create_test_data_db(self.dataset, data_size,
                                        max_num_references=max_num_references)
        print('computing egonet')
        self.plot_factory.host = 'localhost'
        self.plot_factory.port = 27017
        actual = self.plot_factory.plot_egonet(self.dataset, user, depth)

        # Check that every vertex is connected to at least one edge
        vertices = actual['data'][1]
        vertices = np.vstack((vertices['x'], vertices['y'])).T
        edges = actual['data'][0]
        edges = np.vstack((edges['x'], edges['y'])).T
        vertices = {tuple(coord) for coord in vertices if not np.isnan(coord).all()}
        edges = {tuple(coord) for coord in edges if not np.isnan(coord).all()}
        self.assertEqual(edges, vertices)

    def test_no_text(self):
        # Checks it returns the whole thing if the user is not present
        self.plot_factory.simplification = 'backbone'
        self.plot_factory.threshold = 0.232
        self.plot_factory.cache_dir = None
        data_size = 1000
        max_num_references = 10
        user = 'test_user'
        depth = 1
        test_data = create_test_data_db(self.dataset, data_size,
                                        max_num_references=max_num_references)
        print('computing egonet')
        self.plot_factory.host = 'localhost'
        self.plot_factory.port = 27017
        actual = self.plot_factory.plot_egonet(self.dataset, user, depth)

        # Check that every text is different from %text
        self.assertEqual(actual['data'][1]['x'].shape[0], len(set(actual['data'][1]['text'])))

    def test_legitimacy(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data_db(dataset=self.dataset,
                                        day_range=day_range)

        self.plot_factory.unit = 'day'
        self.plot_factory.bin_size = 20 + day_range
        actual = self.plot_factory.get_legitimacy(self.dataset)
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id'])['legitimacy'].sum()

        expected = expected.to_frame()
        expected.columns = actual.columns
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False)

    def test_reputation(self):
        # compute reputation as the amount of referenced tweets attained by each user
        data_size = 100
        day_range = 10
        max_num_references = 20
        test_data = create_test_data_db(dataset=self.dataset,
                                        day_range=day_range, data_size=data_size, max_num_references=max_num_references)

        self.plot_factory.host = 'localhost'
        self.plot_factory.port = 27017
        actual = self.plot_factory.get_reputation(self.dataset)
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.cumsum(axis=1)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True,
                                      check_index_type=False, check_column_type=False)

    def test__get_legitimacy_per_time_1(self):
        # compute legitimacy per time as the amount of referenced tweets attained by each user by unit of time
        day_range = 10
        test_data = create_test_data_db(self.dataset,
                                        day_range=day_range)

        self.plot_factory.unit = 'day'
        self.plot_factory.bin_size = 20 + day_range
        actual = self.plot_factory._get_legitimacy_per_time(self.dataset)
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
        test_data = create_test_data_db(dataset=self.dataset,
                                        day_range=day_range)
        actual = self.plot_factory._get_legitimacy_per_time(self.dataset)
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
        test_data = create_test_data_db(dataset=self.dataset,
                                        day_range=day_range)
        actual = self.plot_factory.get_status(self.dataset)
        expected = pd.DataFrame({'author_id': [t['author']['id'] for t in test_data],
                                 'date': [t['created_at'].date() for t in test_data],
                                 'legitimacy': [len(t['referenced_tweets']) for t in test_data]})

        expected = expected.groupby(['author_id', 'date'])['legitimacy'].sum().to_frame()
        expected = expected.reset_index().pivot(index='author_id', columns='date', values='legitimacy')
        expected.columns = pd.DatetimeIndex(expected.columns)
        expected = expected.cumsum(axis=1)
        expected = expected.apply(lambda x: x.argsort())
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_like=True, check_index_type=False,
                                      check_column_type=False)

    def test_egonet_date(self):
        data_size = 100
        day_range = 10
        max_num_references = 20
        start_date = datetime.fromisoformat("2019-01-01T00:00:00Z")
        end_date = start_date
        test_data = create_test_data_db(dataset=self.dataset,
                                        day_range=day_range, data_size=data_size, max_num_references=max_num_references)

        authors = self.plot_factory._get_authors(self.dataset, start_date=start_date, end_date=end_date)
        references = self.plot_factory._get_references(self.dataset, start_date=start_date, end_date=end_date)
        self.assertEqual(len(authors), 0)
        self.assertEqual(len(references), 0)

    def test_egonet_date_2(self):
        data_size = 100
        day_range = 10
        max_num_references = 20
        start_date = datetime.fromisoformat("2019-01-01T00:00:00Z")
        end_date = datetime.fromisoformat("2019-01-01T00:00:00Z") + timedelta(days=1)

        test_data = create_test_data_db(dataset=self.dataset,
                                        day_range=day_range, data_size=data_size, max_num_references=max_num_references)
        authors = self.plot_factory._get_authors(self.dataset, start_date=start_date, end_date=end_date)
        authors = authors.sort_values('author_id').reset_index(drop=True)
        expected_authors = {}
        expected_references = []
        for tweet in test_data:
            if tweet['created_at'] <= start_date or tweet['created_at'] < end_date:
                expected_authors[tweet['author']['id']] = [tweet['author']['id'], tweet['author']['username'],
                                                           tweet['author']['remiss_metadata']['is_usual_suspect'],
                                                           tweet['author']['remiss_metadata']['party'],
                                                           tweet['author']['public_metrics']['tweet_count'],
                                                           tweet['author']['public_metrics']['followers_count'],
                                                           tweet['author']['public_metrics']['following_count']]

                for referenced_tweet in tweet['referenced_tweets']:
                    expected_references.append([tweet['author']['id'], referenced_tweet['author']['id']])

        expected_authors = pd.DataFrame(expected_authors.values(),
                                        columns=['author_id', 'username', 'is_usual_suspect', 'party', 'num_tweets',
                                                 'num_followers', 'num_following'])
        expected_authors = expected_authors.drop_duplicates().sort_values('author_id').reset_index(drop=True)
        authors = authors.sort_values('author_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(expected_authors, authors, check_dtype=False, check_like=True,
                                      check_index_type=False)

        references = self.plot_factory._get_references(self.dataset, start_date=start_date, end_date=end_date)
        references = references.sort_values(['source', 'target']).reset_index(drop=True)
        references = references[['source', 'target']]
        expected_references = pd.DataFrame(expected_references, columns=['source', 'target'])
        expected_references = expected_references.drop_duplicates().sort_values(['source', 'target']).reset_index(
            drop=True)
        pd.testing.assert_frame_equal(expected_references, references, check_dtype=False, check_like=True,
                                      check_index_type=False)


def create_test_data_db(dataset='unit_test_remiss', data_size=100, day_range=10,
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
                                "remiss_metadata": {"party": party, "is_usual_suspect": is_usual_suspect},
                                'public_metrics': {'tweet_count': 1, 'followers_count': 100, 'following_count': 10}},
                     "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                     'referenced_tweets': referenced_tweets,
                     'created_at': created_at}
            test_data.append(tweet)

    client = MongoClient('localhost', 27017)
    client.drop_database(dataset)
    database = client.get_database(dataset)
    collection = database.get_collection('raw')
    collection.insert_many(test_data)

    return test_data
