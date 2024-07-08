import pickle
import unittest
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

import igraph as ig

from models.propagation import PropagationDatasetGenerator, PropagationCascadeModel


class PropagationModelsTestCase(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     populate_test_database('test_dataset')
    #     populate_test_database('test_dataset_small', small=True)

    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database('test_dataset')
    #     delete_test_database('test_dataset_small')

    def setUp(self):
        self.dataset_generator = PropagationDatasetGenerator('test_dataset')
        self.dataset = 'test_dataset'
        self.dataset_small = 'test_dataset_small'
        self.cache_dir = Path('tmp/cache_propagation_2')

    def test_prepare_propagation_dataset(self):
        dataset = self.dataset
        features = self.dataset_generator.generate_propagation_dataset()

        features.to_csv(self.cache_dir / f'{dataset}-features.csv')
        # num_samples = np.minimum(100, features.shape[0])
        # sns.pairplot(features.sample(num_samples), hue='propagated', diag_kind='kde')
        # plt.savefig('tmp/cache_propagation_2/pairplot.png')

    def test_fit(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        model.fit(features)
        with open(self.cache_dir / f'{dataset}-model.pkl', 'wb') as f:
            pickle.dump(model, f)

    def test_fit_2(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        X, y = features.drop(columns=['propagated']), features['propagated']
        model.fit(X, y)

    def test_fit_3(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
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
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        X = features.drop(columns=['propagated'])
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': '1298573780961370112', 'author_id': '18202143'}
        model.dataset_generator = self.dataset_generator
        cascade = model.generate_cascade(sample)
        fig, ax = plt.subplots()

        ig.plot(cascade, target=ax)
        plt.show()


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
        self.assertEqual(features.shape[1]-1, tweet_features.shape[1])
        model = PropagationCascadeModel()
        with open(self.cache_dir / f'{self.dataset_generator.dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        model_expected_feature_columns = model.pipeline.feature_names_in_
        self.assertEqual(list(tweet_features.columns), list(model_expected_feature_columns))
        pred = model.predict(tweet_features)