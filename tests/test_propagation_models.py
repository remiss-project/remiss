import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import igraph as ig

from models.propagation import PropagationDatasetGenerator, PropagationCascadeModel


class PropagationModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = 'test_dataset_2'

        self.dataset_generator = PropagationDatasetGenerator(self.dataset)
        self.dataset_small = 'test_dataset_small'
        self.cache_dir = Path('tmp/cache_propagation')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def test_prepare_propagation_dataset(self):
        dataset = self.dataset
        features = self.dataset_generator.generate_propagation_dataset()

        features.to_csv(self.cache_dir / f'{dataset}-features.csv')
        # num_samples = np.minimum(100, features.shape[0])
        # sns.pairplot(features.sample(num_samples), hue='propagated', diag_kind='kde')
        # plt.savefig('tmp/cache_propagation_2/pairplot.png')

    @unittest.skip("Slow")
    def test_fit(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        features = features.head(1000)
        model.fit(features)
        with open(self.cache_dir / f'{dataset}-model.pkl', 'wb') as f:
            pickle.dump(model, f)

    @unittest.skip("Slow")
    def test_fit_2(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        features = features.head(1000)
        X, y = features.drop(columns=['propagated']), features['propagated']
        model.fit(X, y)

    @unittest.skip("Slow")
    def test_fit_3(self):
        model = PropagationCascadeModel()
        dataset = self.dataset_small
        model.fit(dataset)

    def test_predict(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        X = features.drop(columns=['propagated'])
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X)
        print(y_pred)

    def test_predict_proba(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        X = features.drop(columns=['propagated'])
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict_proba(X)
        print(y_pred)

    def test_generate_cascade(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': '1298573780961370112', 'author_id': '18202143'}
        model.dataset_generator = self.dataset_generator
        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        fig.show()

    def test_generate_cascade_2(self):
        class MockModel:
            def __init__(self, limit=1):
                self.current = 0
                self.limit = limit

            def predict(self, X):
                pred = np.zeros(X.shape[0], dtype=int)
                remaining = self.limit - self.current
                if remaining > 0:
                    available = np.minimum(remaining, X.shape[0])
                    pred[:available] = 1
                    self.current += available
                    np.random.shuffle(pred)
                return pred

        dataset = self.dataset
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': '1298573780961370112', 'author_id': '18202143'}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=0)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        fig.show()
        self.assertEqual(len(cascade.vs), 45)
        self.assertEqual(len(cascade.es), 40)

    def test_generate_cascade_3(self):
        class MockModel:
            def __init__(self, limit=1):
                self.current = 0
                self.limit = limit

            def predict(self, X):
                pred = np.zeros(X.shape[0], dtype=int)
                remaining = self.limit - self.current
                if remaining > 0:
                    available = np.minimum(remaining, X.shape[0])
                    pred[:available] = 1
                    self.current += available
                    np.random.shuffle(pred)
                return pred

        dataset = self.dataset
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': '1298573780961370112', 'author_id': '18202143'}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=1)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        fig.show()
        self.assertEqual(len(cascade.vs), 46)
        self.assertEqual(len(cascade.es), 41)

    def test_generate_cascade_4(self):
        class MockModel:
            def __init__(self, limit=1):
                self.current = 0
                self.limit = limit

            def predict(self, X):
                pred = np.zeros(X.shape[0], dtype=int)
                remaining = self.limit - self.current
                if remaining > 0:
                    available = np.minimum(remaining, X.shape[0])
                    pred[:available] = 1
                    self.current += available
                    np.random.shuffle(pred)
                return pred

        dataset = self.dataset
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': '1298573780961370112', 'author_id': '18202143'}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=2)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        fig.show()
        self.assertEqual(len(cascade.vs), 47)
        self.assertEqual(len(cascade.es), 42)

    def test_generate_cascade_4(self):
        class MockModel:
            def __init__(self, limit=1):
                self.current = 0
                self.limit = limit

            def predict(self, X):
                pred = np.zeros(X.shape[0], dtype=int)
                remaining = self.limit - self.current
                if remaining > 0:
                    available = np.minimum(remaining, X.shape[0])
                    pred[:available] = 1
                    self.current += available
                    np.random.shuffle(pred)
                return pred

        dataset = self.dataset
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': '1298573780961370112', 'author_id': '18202143'}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=10)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        fig.show()
        fig, ax = plt.subplots()
        ig.plot(cascade, target=ax, node_size=2)
        plt.show()
        self.assertEqual(len(cascade.vs), 55)
        self.assertEqual(len(cascade.es), 50)

    def test_get_cascade(self):
        cascade = self.dataset_generator.get_cascade('1298573780961370112', None)
        fig, ax = plt.subplots()

        ig.plot(cascade, target=ax)
        plt.show()

    def test_get_features_for_cascade(self):
        conversation_id = '1298573780961370112'
        user_id = '18202143'
        neighbour = self.dataset_generator.get_neighbours(user_id)[0]
        cascade = self.dataset_generator.get_cascade(conversation_id, user_id)
        tweet_features = self.dataset_generator.get_features_for(conversation_id, user_id, neighbour)
        features = self.dataset_generator.generate_propagation_dataset()
        self.assertEqual(features.shape[1] - 1, tweet_features.shape[1])
        model = PropagationCascadeModel()
        with open(self.cache_dir / f'{self.dataset_generator.dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        model_expected_feature_columns = model.pipeline.feature_names_in_
        self.assertEqual(list(tweet_features.columns), list(model_expected_feature_columns))
        pred = model.predict(tweet_features)
