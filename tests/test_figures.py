import cProfile
import random
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import igraph as ig
import pandas as pd
from plotly.graph_objs import Figure
from pymongo import MongoClient

from figures import EgonetPlotFactory
from figures import MongoPlotFactory
from figures import TweetUserPlotFactory


class TestMongoPlotFactory(unittest.TestCase):
    def setUp(self):
        self.mongo_plot = MongoPlotFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_remiss')
        self.database = self.client.get_database('test_remiss')
        self.collection = self.database.get_collection('test_collection')
        test_data = [{"id": 0, "created_at": {"$date": "2019-01-01T23:20:00Z"},
                      "author": {"username": "TEST_USER_0", "id": 0,
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": 1, "created_at": {"$date": "2019-01-02T23:20:00Z"},
                      "author": {"username": "TEST_USER_1", "id": 1,
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "quoted"}]},
                     {"id": 2, "created_at": {"$date": "2019-01-03T23:20:00Z"},
                      "author": {"username": "TEST_USER_2", "id": 2,
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

    def tearDown(self) -> None:
        self.collection.drop()
        self.client.drop_database('test_remiss')
        self.client.close()

    def test_get_date_range(self):
        # Mock MongoClient and database
        mock_date = Mock()
        mock_date.date.return_value = 'test_date'
        mock_collection = Mock()
        mock_collection.find_one.return_value = {'created_at': mock_date}
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        with patch('figures.MongoClient', return_value=mock_client):
            date_range = self.mongo_plot.get_date_range("test_collection")
            self.assertEqual(date_range, ('test_date', 'test_date'))

    def test_get_hashtag_freqs(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_collection.aggregate.return_value = [
            {'_id': 'test_hashtag1', 'count': 1},
            {'_id': 'test_hashtag2', 'count': 2},
            {'_id': 'test_hashtag3', 'count': 3},
        ]
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        with patch('figures.MongoClient', return_value=mock_client):
            hashtag_freqs = self.mongo_plot.get_hashtag_freqs("test_collection")
            self.assertEqual(hashtag_freqs, [(x['_id'], x['count']) for x in mock_collection.aggregate.return_value])

    def test_get_hashtag_freqs_2(self):
        expected = [('test_hashtag', 1)]
        actual = self.mongo_plot.get_hashtag_freqs("test_collection")
        self.assertEqual(expected, actual)

    def test_get_users(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_collection.distinct.return_value = ['test_user1', 'test_user2']
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database
        with patch('figures.MongoClient', return_value=mock_client):
            users = self.mongo_plot.get_users("test_collection")
            self.assertEqual(users, [str(x) for x in mock_collection.distinct.return_value])

    def test_get_users_2(self):
        expected = ['TEST_USER_0', 'TEST_USER_1', 'TEST_USER_2']
        actual = self.mongo_plot.get_users("test_collection")
        self.assertEqual(expected, actual)

    def test_get_user_id(self):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_client = Mock()
        mock_client.get_database.return_value = mock_database

        # Mock find_one
        mock_collection.find_one.return_value = {
            'author': {'username': 'test_username', 'id': 'test_id'}
        }

        with patch('figures.MongoClient', return_value=mock_client):
            user_id = self.mongo_plot.get_user_id("test_collection", "test_username")
            self.assertEqual(user_id, 'test_id')

    def test_get_user_id_2(self):
        expected = 0
        actual = self.mongo_plot.get_user_id("test_collection", "TEST_USER_0")
        self.assertEqual(expected, actual)

    def test_available_datasets(self):
        # Mock MongoClient
        mock_client = Mock()
        mock_client.get_database.return_value.list_collection_names.return_value = ['collection1', 'collection2']
        with patch('figures.MongoClient', return_value=mock_client):
            datasets = self.mongo_plot.available_datasets
            self.assertEqual(datasets, mock_client.get_database.return_value.list_collection_names.return_value)

    def test_available_datasets_2(self):
        expected = ['test_collection']
        actual = self.mongo_plot.available_datasets
        self.assertEqual(expected, actual)


class TestTweetUserPlotFactory(unittest.TestCase):
    def setUp(self):
        self.tweet_user_plot = TweetUserPlotFactory()
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_remiss')
        self.database = self.client.get_database('test_remiss')
        self.collection = self.database.get_collection('test_collection')
        test_data = [{"id": 0, "created_at": datetime.fromisoformat("2019-01-01T23:20:00Z"),
                      "author": {"username": "TEST_USER_0", "id": 0,
                                 "remiss_metadata": {"party": "PSOE", "is_usual_suspect": False}},
                      "entities": {"hashtags": [{"tag": "test_hashtag"}]},
                      "referenced_tweets": []},
                     {"id": 1, "created_at": datetime.fromisoformat("2019-01-02T23:20:00Z"),
                      "author": {"username": "TEST_USER_1", "id": 1,
                                 "remiss_metadata": {"party": None, "is_usual_suspect": False}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "quoted"}]},
                     {"id": 2, "created_at": datetime.fromisoformat("2019-01-03T23:20:00Z"),
                      "author": {"username": "TEST_USER_2", "id": 2,
                                 "remiss_metadata": {"party": "VOX", "is_usual_suspect": True}},
                      "entities": {"hashtags": []},
                      "referenced_tweets": [{"id": 1, "type": "retweeted"}]}]
        self.collection.insert_many(test_data)

    def test_plot_tweet_series(self):
        # Mock _get_counts
        self.tweet_user_plot._get_count_area_plot = Mock(return_value='mocked_plot_data')

        collection = 'test_collection'
        hashtag = 'test_hashtag'
        start_time = '2023-01-01'
        end_time = '2023-01-31'
        unit = 'day'
        bin_size = 1
        expected_pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]

        result = self.tweet_user_plot.plot_tweet_series(collection, hashtag, start_time, end_time, unit, bin_size)

        self.assertEqual(result, 'mocked_plot_data')
        self.tweet_user_plot._get_count_area_plot.assert_called_with(
            expected_pipeline, collection, hashtag, start_time, end_time, unit, bin_size
        )

    def test_plot_tweet_series_2(self):
        actual = self.tweet_user_plot.plot_tweet_series('test_collection', None, None, None)
        self.assertEqual([datetime(2019, 1, 1, 0, 0),
                          datetime(2019, 1, 2, 0, 0),
                          datetime(2019, 1, 3, 0, 0)], list(actual['data'][0]['x']))
        self.assertEqual([1, 1, 1], list(actual['data'][0]['y']))

    def test_plot_user_series(self):
        # Mock _get_counts
        self.tweet_user_plot._get_count_area_plot = Mock(return_value='mocked_plot_data')

        collection = 'test_collection'
        hashtag = 'test_hashtag'
        start_time = '2023-01-01'
        end_time = '2023-01-31'
        unit = 'day'
        bin_size = 1
        expected_pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "users": {'$addToSet': "$author.username"}
            }
            },
            {'$project': {'count': {'$size': '$users'}}},
            {'$sort': {'_id': 1}}
        ]
        result = self.tweet_user_plot.plot_user_series(collection, hashtag, start_time, end_time, unit, bin_size)

        self.assertEqual(result, 'mocked_plot_data')
        self.tweet_user_plot._get_count_area_plot.assert_called_with(
            expected_pipeline, collection, hashtag, start_time, end_time, unit, bin_size
        )

    def test_plot_user_series_2(self):
        actual = self.tweet_user_plot.plot_user_series('test_collection', None, None, None)
        self.assertEqual([datetime(2019, 1, 1, 0, 0),
                          datetime(2019, 1, 2, 0, 0),
                          datetime(2019, 1, 3, 0, 0)], list(actual['data'][0]['x']))
        self.assertEqual([1, 1, 1], list(actual['data'][0]['y']))

    @patch('figures.MongoClient')
    def test__get_counts(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()
        mock_database = Mock()
        mock_database.get_collection.return_value = mock_collection
        mock_mongo_client.return_value.get_database.return_value = mock_database

        # Mock aggregate_pandas_all
        mock_collection.aggregate_pandas_all.return_value = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                                                                  datetime(2023, 1, 2),
                                                                                  datetime(2023, 1, 3)],
                                                                          'count': [1, 2, 3]})

        # Mock _add_filters
        self.tweet_user_plot._add_filters = Mock(return_value='mocked_pipeline')

        collection = 'test_collection'
        hashtag = 'test_hashtag'
        start_time = '2023-01-01'
        end_time = '2023-01-31'
        unit = 'day'
        bin_size = 1

        expected_pipeline = [
            {'$group': {
                "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
                "count": {'$count': {}}}},
            {'$sort': {'_id': 1}}
        ]
        result = self.tweet_user_plot._get_count_area_plot(
            expected_pipeline, collection, hashtag, start_time, end_time
        )

        self.assertIsInstance(result, Figure)

        mock_mongo_client.assert_called_with(self.tweet_user_plot.host, self.tweet_user_plot.port)
        mock_database.get_collection.assert_called_with(collection)

    def test__add_filters(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        user_type = 'normal'
        expected_pipeline = [{'$match': {'created_at': {'$lte': end_time}}},
                             {'$match': {'created_at': {'$gte': start_time}}},
                             {'$match': {'entities.hashtags.tag': hashtag}},
                             {'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': None}}]

        result = self.tweet_user_plot._add_filters([], hashtag, start_time, end_time, user_type)

        self.assertEqual(result, expected_pipeline)

    def test__add_filters_normal(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': None}}]
        result = self.tweet_user_plot._add_filters([], hashtag=None, start_time=None, end_time=None,
                                                   user_type='normal')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_suspect(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                         'author.remiss_metadata.party': None}}]
        result = self.tweet_user_plot._add_filters([], hashtag=None, start_time=None, end_time=None,
                                                   user_type='suspect')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_politician(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': False,
                                         'author.remiss_metadata.party': {'$ne': None}}}]
        result = self.tweet_user_plot._add_filters([], hashtag=None, start_time=None, end_time=None,
                                                   user_type='politician')
        self.assertEqual(result, expected_pipeline)

    def test__add_filters_suspect_politician(self):
        expected_pipeline = [{'$match': {'author.remiss_metadata.is_usual_suspect': True,
                                         'author.remiss_metadata.party': {'$ne': None}}}]
        result = self.tweet_user_plot._add_filters([], hashtag=None, start_time=None, end_time=None,
                                                   user_type='suspect_politician')
        self.assertEqual(result, expected_pipeline)

    def test__get_count_data(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []

        collection = Mock()
        test_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          datetime(2023, 1, 3)],
                                  'count': [1, 2, 3]})
        collection.aggregate_pandas_all.return_value = test_data
        actual = self.tweet_user_plot._get_count_data(pipeline, hashtag, start_time, end_time, collection)
        test_data = test_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([test_data] * 4, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test__get_count_data_empty(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []

        collection = Mock()
        good_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          datetime(2023, 1, 3)],
                                  'count': [1, 2, 3]})
        bad_data = pd.DataFrame({'_id': [],
                                 'count': []})

        class TestData:
            def __init__(self):
                self.i = 0

            def get_data(self, pipeline, schema):
                if self.i < 2:
                    data = good_data
                else:
                    data = bad_data
                self.i += 1
                return data

        test_data = TestData()
        collection.aggregate_pandas_all = test_data.get_data
        actual = self.tweet_user_plot._get_count_data(pipeline, hashtag, start_time, end_time, collection)
        good_data = good_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        bad_data = bad_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([good_data] * 2 + [bad_data] * 2, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        pd.testing.assert_frame_equal(actual, expected)

    def test__get_count_data_index_unmatched(self):
        hashtag = 'test_hashtag'
        start_time = pd.to_datetime('2023-01-01')
        end_time = pd.to_datetime('2023-01-31')
        pipeline = []

        collection = Mock()
        good_data = pd.DataFrame({'_id': [datetime(2023, 1, 1),
                                          datetime(2023, 1, 2),
                                          ],
                                  'count': [1, 2]})
        bad_data = pd.DataFrame({'_id': [
            datetime(2023, 1, 2),
            datetime(2023, 1, 3)],
            'count': [2, 3]})

        class TestData:
            def __init__(self):
                self.i = 0

            def get_data(self, pipeline, schema):
                if self.i < 2:
                    data = good_data
                else:
                    data = bad_data
                self.i += 1
                return data

        test_data = TestData()
        collection.aggregate_pandas_all = test_data.get_data
        actual = self.tweet_user_plot._get_count_data(pipeline, hashtag, start_time, end_time, collection)
        good_data = good_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        bad_data = bad_data.copy().rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time')
        expected = pd.concat([good_data] * 2 + [bad_data] * 2, axis=1)
        expected.columns = ['Normal', 'Usual suspect', 'Politician', 'Usual suspect politician']
        pd.testing.assert_frame_equal(actual, expected)


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
            if len(pipeline) == 3:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4]})
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

        actual, layout = self.egonet_plot.get_egonet(collection, user, depth)

        self.assertEqual({1, 2}, set(actual.vs['id_']))
        edges = {(actual.vs[s]['id_'], actual.vs[t]['id_']) for s, t in actual.get_edgelist()}
        self.assertEqual({(1, 2)}, edges)

    @patch('figures.MongoClient')
    def test_get_egonet_2(self, mock_mongo_client):
        # Checks it returns the whole thing if the user is not present

        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if len(pipeline) == 3:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4]})
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

        actual, layout = self.egonet_plot.get_egonet(collection, user, depth)

        self.assertEqual({1, 2}, set(actual.vs['id_']))
        edges = {(actual.vs[s]['id_'], actual.vs[t]['id_']) for s, t in actual.get_edgelist()}
        self.assertEqual({(3, 4)}, edges)

    @patch('figures.MongoClient')
    def test_compute_hidden_network(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if len(pipeline) == 3:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 1]})
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

        actual, layout = self.egonet_plot.compute_hidden_network(collection)

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
            if len(pipeline) == 3:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4]})
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

        actual, layout = self.egonet_plot.compute_hidden_network(collection)

        self.assertEqual({1, 2, 3, 4}, set(actual.vs['id_']))
        edges = {(actual.vs[s]['id_'], actual.vs[t]['id_']) for s, t in actual.get_edgelist()}
        self.assertEqual({(2, 3), (1, 2), (3, 4)}, edges)

    @patch('figures.MongoClient')
    def test_compute_hidden_network_count(self, mock_mongo_client):
        # Mock MongoClient and database
        mock_collection = Mock()

        def aggregate_pandas_all(pipeline):
            if len(pipeline) == 3:
                # its edges
                edges = pd.DataFrame({'source': [1, 2, 3],
                                      'target': [2, 3, 4]})
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

        graph, layout = self.egonet_plot.compute_hidden_network(collection)

        actual = graph.vs['count']
        expected = [1, 1, 1, 1]
        self.assertEqual(expected, actual)

    def test__get_references(self):
        test_data = []
        expected_edges = pd.DataFrame({'source': [1, 2, 3],
                                       'target': [2, 3, 4]})
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
        self.assertEqual(actual['weight'].to_list(), [1, 1, 1])
        # self.assertEqual(actual['weight_norm'].sum(), 1)
        self.assertEqual(actual['weight_inv'].to_list(), [1, 1, 1])
        self.assertEqual(actual['source'], [1, 2, 3])
        self.assertEqual(actual['target'], [2, 3, 4])

    def test_plot_egonet(self):
        # Mock get_egonet
        network = ig.Graph.GRG(8, 0.2)
        network.vs['id_'] = [0, 1, 2, 3, 4, 5, 6, 7]
        network.vs['username'] = ['TEST_USER_0', 'TEST_USER_1', 'TEST_USER_2', 'TEST_USER_3', 'TEST_USER_4',
                                  'TEST_USER_5', 'TEST_USER_6', 'TEST_USER_7']
        network.vs['party'] = ['PSOE', None, 'VOX', None, 'PSOE', None, 'VOX', None]
        network.vs['is_usual_suspect'] = [False, False, False, False, True, True, True, True]
        self.egonet_plot.get_egonet = Mock(return_value=(network, None))

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

        client = MongoClient('localhost', 27017)
        client.drop_database('test_remiss')
        database = client.get_database('test_remiss')
        collection = database.get_collection('test_collection')
        print(f'storing test data {total_referenced_tweets}')
        collection.insert_many(test_data)

        collection = 'test_collection'
        authors, references = self.egonet_plot._get_authors_and_references(collection)
        self.assertEqual(data_size // 2, len(authors))
        self.assertEqual(total_referenced_tweets, len(references))
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
        actual, layout = self.egonet_plot.get_egonet(collection, user, depth)
        self.assertEqual(data_size // 2, actual.vcount())
        self.assertEqual(total_referenced_tweets, actual.ecount())
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
        actual, layout = self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time_no_cache = end_time - start_time
        print(f'took {total_time_no_cache} no cache')
        self.assertLessEqual(total_time_no_cache, 60)
        self.assertTrue(Path('/tmp/remiss_cache/test_collection.graphmlz').exists())
        self.assertTrue(Path('/tmp/remiss_cache/test_collection.feather').exists())
        start_time = time.time()
        actual, layout = self.egonet_plot.get_egonet(collection, user, depth)
        end_time = time.time()
        total_time = end_time - start_time
        print(f'took {total_time}')
        self.assertLess(total_time, total_time_no_cache)
        self.assertTrue(actual.get_edgelist(),
                        ig.Graph.Read_GraphMLz('/tmp/remiss_cache/test_collection.graphmlz').get_edgelist())
        self.assertEqual(actual.vcount(),
                         pd.read_feather('/tmp/remiss_cache/test_collection.feather').shape[0])


if __name__ == '__main__':
    unittest.main()
