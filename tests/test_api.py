import json
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
from pymongo import MongoClient

from api import create_app
from preprocessor import PropagationPreprocessor
from propagation import NetworkMetrics, DiffusionMetrics, Egonet
from propagation.histogram import Histogram
from tests.conftest import delete_test_database


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dataset = 'new_dataset'
        self.tmp_dataset = 'tmp_dataset'
        self.num_samples = 100

    def test_store_raw(self):
        with open('test_resources/Openarms.preprocessed.jsonl', 'r') as f:
            expected_data = [json.loads(line) for line in f.readlines()[:self.num_samples]]
        preprocessor = PropagationPreprocessor(dataset=self.tmp_dataset, data=expected_data)
        preprocessor.store_raw_data()
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        actual_data = list(collection.find())

        # Check all dates are dates instead of strings
        def assert_date_format(tweet):
            date_fields = {'created_at', 'editable_until', 'retrieved_at'}
            for field, value in tweet.items():
                if field in date_fields:
                    assert isinstance(value, pd.Timestamp) or isinstance(value, datetime)

                elif isinstance(value, dict):
                    assert_date_format(value)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, dict):
                            assert_date_format(v)

        for tweet in actual_data:
            assert_date_format(tweet)

        cast_datetimes_to_timestamps(actual_data)
        cast_datetimes_to_timestamps(expected_data)

        # Check all data is the same
        assert expected_data == actual_data

    def test_preprocess_network_metrics(self):
        with open('test_resources/Openarms.preprocessed.jsonl', 'r') as f:
            data = [json.loads(line) for line in f.readlines()[:self.num_samples]]
        preprocessor = PropagationPreprocessor(dataset=self.tmp_dataset, data=data)
        preprocessor.process()
        network_metrics = NetworkMetrics(host=preprocessor.host, port=preprocessor.port,
                                         reference_types=preprocessor.reference_types)
        expected_legitimacy = network_metrics.compute_legitimacy(self.tmp_dataset).sort_index()
        expected_reputation = network_metrics.compute_reputation(self.tmp_dataset).sort_index()
        expected_status = network_metrics.compute_status(self.tmp_dataset).sort_index()

        network_metrics = NetworkMetrics(host=preprocessor.host, port=preprocessor.port,
                                         reference_types=preprocessor.reference_types)

        actual_legitimacy = network_metrics.get_legitimacy(self.tmp_dataset).sort_index()
        actual_reputation = network_metrics.get_reputation(self.tmp_dataset).sort_index()
        actual_status = network_metrics.get_status(self.tmp_dataset).sort_index()

        pd.testing.assert_series_equal(expected_legitimacy, actual_legitimacy, check_dtype=False)
        pd.testing.assert_frame_equal(expected_reputation, actual_reputation, check_dtype=False, check_index_type=False,
                                      check_column_type=False)
        pd.testing.assert_frame_equal(expected_status, actual_status, check_dtype=False, check_index_type=False,
                                      check_column_type=False)

    def test_preprocess_diffusion_metrics(self):
        with open('test_resources/Openarms.preprocessed.jsonl', 'r') as f:
            data = [json.loads(line) for line in f.readlines()[:self.num_samples]]
        preprocessor = PropagationPreprocessor(dataset=self.tmp_dataset, data=data)
        preprocessor.process()
        diffusion_metrics = DiffusionMetrics(host=preprocessor.host, port=preprocessor.port,
                                             reference_types=preprocessor.reference_types)

        conversation_ids = diffusion_metrics.get_conversation_ids(self.tmp_dataset)
        expected = {}
        for conversation_id in conversation_ids['conversation_id']:
            graph = diffusion_metrics.compute_propagation_tree(self.tmp_dataset, conversation_id)
            size_over_time = diffusion_metrics.compute_size_over_time(graph)
            depth_over_time = diffusion_metrics.compute_depth_over_time(graph)
            max_breadth_over_time = diffusion_metrics.compute_max_breadth_over_time(graph)
            structural_virality_over_time = diffusion_metrics.compute_structural_virality_over_time(graph)
            expected[conversation_id] = {'graph': graph, 'size_over_time': size_over_time,
                                         'depth_over_time': depth_over_time,
                                         'max_breadth_over_time': max_breadth_over_time,
                                         'structural_virality_over_time': structural_virality_over_time}
        diffusion_metrics = DiffusionMetrics(host=preprocessor.host, port=preprocessor.port,
                                             reference_types=preprocessor.reference_types)

        actual = {}
        for conversation_id in conversation_ids['conversation_id']:
            graph = diffusion_metrics.get_propagation_tree(self.tmp_dataset, conversation_id)
            size_over_time = diffusion_metrics.get_size_over_time(self.tmp_dataset, conversation_id)
            depth_over_time = diffusion_metrics.get_depth_over_time(self.tmp_dataset, conversation_id)
            max_breadth_over_time = diffusion_metrics.get_max_breadth_over_time(self.tmp_dataset, conversation_id)
            structural_virality_over_time = diffusion_metrics.get_structural_virality_over_time(self.tmp_dataset,
                                                                                                conversation_id)
            actual[conversation_id] = {'graph': graph, 'size_over_time': size_over_time,
                                       'depth_over_time': depth_over_time,
                                       'max_breadth_over_time': max_breadth_over_time,
                                       'structural_virality_over_time': structural_virality_over_time}

        for conversation_id in conversation_ids['conversation_id']:
            expected_graph = expected[conversation_id]['graph']
            expected_size_over_time = expected[conversation_id]['size_over_time']
            expected_depth_over_time = expected[conversation_id]['depth_over_time']
            expected_max_breadth_over_time = expected[conversation_id]['max_breadth_over_time']
            expected_structural_virality_over_time = expected[conversation_id]['structural_virality_over_time']

            actual_graph = actual[conversation_id]['graph']
            actual_size_over_time = actual[conversation_id]['size_over_time']
            actual_depth_over_time = actual[conversation_id]['depth_over_time']
            actual_max_breadth_over_time = actual[conversation_id]['max_breadth_over_time']
            actual_structural_virality_over_time = actual[conversation_id]['structural_virality_over_time']

            assert expected_graph.isomorphic(actual_graph)
            pd.testing.assert_series_equal(expected_size_over_time, actual_size_over_time, check_dtype=False)
            pd.testing.assert_series_equal(expected_depth_over_time, actual_depth_over_time, check_dtype=False)
            pd.testing.assert_series_equal(expected_max_breadth_over_time, actual_max_breadth_over_time,
                                           check_dtype=False)
            pd.testing.assert_series_equal(expected_structural_virality_over_time, actual_structural_virality_over_time,
                                           check_dtype=False)

    def test_egonet(self):
        with open('test_resources/Openarms.preprocessed.jsonl', 'r') as f:
            data = [json.loads(line) for line in f.readlines()[:self.num_samples]]
        preprocessor = PropagationPreprocessor(dataset=self.tmp_dataset, data=data)
        preprocessor.process()
        egonet = Egonet(host=preprocessor.host, port=preprocessor.port,
                        reference_types=preprocessor.reference_types)

        actual_hidden_network = egonet.get_hidden_network(self.tmp_dataset)
        actual_backbone = egonet.get_hidden_network_backbone(self.tmp_dataset)

        egonet = Egonet(host=preprocessor.host, port=preprocessor.port,
                        reference_types=preprocessor.reference_types)

        expected_hidden_network = egonet._compute_hidden_network(self.tmp_dataset)

        expected_backbone = egonet._simplify_graph(expected_hidden_network)

        assert expected_hidden_network.isomorphic(actual_hidden_network)
        assert expected_backbone.isomorphic(actual_backbone)

    def test_histograms(self):
        with open('test_resources/Openarms.preprocessed.jsonl', 'r') as f:
            data = [json.loads(line) for line in f.readlines()[:self.num_samples]]
        preprocessor = PropagationPreprocessor(dataset=self.tmp_dataset, data=data)
        preprocessor.process()
        histogram = Histogram(host=preprocessor.host, port=preprocessor.port)
        expected_tweet_histogram = histogram.compute_tweet_histogram(self.tmp_dataset, [], None, None)
        expected_user_histogram = histogram.compute_user_histogram(self.tmp_dataset, [], None, None)
        histogram = Histogram(host=preprocessor.host, port=preprocessor.port)
        actual_tweet_histogram = histogram.load_histogram(self.tmp_dataset, 'tweet')
        actual_user_histogram = histogram.load_histogram(self.tmp_dataset, 'user')
        pd.testing.assert_frame_equal(expected_tweet_histogram, actual_tweet_histogram, check_dtype=False)
        pd.testing.assert_frame_equal(expected_user_histogram, actual_user_histogram, check_dtype=False)


    def test_api(self):
        test_file = Path('test_resources/Openarms.sample.jsonl')
        with open(test_file, 'r') as f:
            expected_data = [json.loads(line) for line in f]

        app = create_app()
        client = app.test_client()
        # make request
        with open(test_file, 'rb') as f:
            response = client.post(f'/process_dataset?db_name={self.tmp_dataset}',
                                   data={'file': (f, test_file.name)},
                                   content_type='multipart/form-data')
        assert response.status_code == 200
        assert response.json == {"message": f"Processed dataset {self.tmp_dataset}"}
        # Check that the data is actually in the db
        client = MongoClient('localhost', 27017)
        database = client.get_database(self.tmp_dataset)
        collection = database.get_collection('raw')
        actual_data = list(collection.find({}, {'_id': 0}))
        cast_datetimes_to_timestamps(actual_data)
        cast_datetimes_to_timestamps(expected_data)
        assert expected_data == actual_data

    def tearDown(self):
        delete_test_database(self.tmp_dataset)


def cast_datetimes_to_timestamps(data):
    for tweet in data:
        cast_tweet_datetimes_to_timestamps(tweet)


def cast_tweet_datetimes_to_timestamps(tweet):
    date_fields = {'created_at', 'editable_until', 'retrieved_at'}
    for field, value in tweet.items():
        if field in date_fields:

            tweet[field] = pd.Timestamp(value).tz_localize(None)

        elif isinstance(value, dict):
            cast_tweet_datetimes_to_timestamps(value)
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, dict):
                    cast_tweet_datetimes_to_timestamps(v)


if __name__ == '__main__':
    unittest.main()
