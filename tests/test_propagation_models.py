import pickle
import unittest
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.propagation import PropagationDatasetGenerator, PropagationCascadeModel


class PropagationModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = 'test_dataset_2'

        self.dataset_generator = PropagationDatasetGenerator(self.dataset)
        self.cache_dir = Path('tmp/cache_propagation')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.test_tweet_id = '1167074391315890176'

    def test_get_available_cascades(self):
        cascades = self.dataset_generator.get_available_cascades()
        self.assertGreater(len(cascades), 2)

    def test_get_rows(self):
        cascades = self.dataset_generator.get_available_cascades()
        rows = self.dataset_generator.get_rows(cascades)
        self.assertGreater(len(rows), 1700)

    def test_generate_propagation_dataset(self):
        dataset = self.dataset_generator.generate_propagation_dataset()
        self.assertEqual(dataset.shape[1], 820)
        self.assertEqual(dataset.shape[0], 2511)

    def test_fetch_tweet_features(self):
        tweets = ['1167084036638027778', '1167083487876263938', '1167082116171141121', '1167081267822772224',
                  '1167081127645011970', '1167080953606524928', '1167080726434603009', '1167080443591544834',
                  '1167080378827509760', '1167080061444546561', '1167079805579419649', '1167078963581194240',
                  '1167078010442436608', '1167076670148370433', '1166410242848165889', '1165197752336289792',
                  '1164609510046150656', '1164521852984811522', '1164249094434504704', '1164199001316569089',
                  '1164157756166852621', '1164140657293836288', '1164129805970878466', '1164120137789771777']
        tweet_features = self.dataset_generator.fetch_tweet_features(tweets)
        self.assertEqual(len(tweet_features), len(tweets))

    def test_fetch_user_features(self):
        users = ['1005545356816474113', '1033714286231740416', '1044030528675090433', '1047520952228216833',
                 '1052919220281987074', '1064067980043186177', '1080272105755893760', '1085590459714539520',
                 '1106824822850490368', '1113508263562235904', '1117506505601896449', '1121337101314863105',
                 '1121824938569216000', '1123645074095714304', '1142093391339118593', '1143993774114254848',
                 '1145353836', '1167078506569883651', '1244846407', '1285099554', '132842161', '1361584274']

        user_features = self.dataset_generator.fetch_user_features(users)
        self.assertEqual(len(user_features), 20)

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

    @unittest.skip('Slow')
    def test_fit_2(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        X, y = features.drop(columns=['propagated']), features['propagated']
        model.fit(X, y)

    @unittest.skip('Slow')
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

    @unittest.skip('Slow')
    def test_generate_cascade(self):
        model = PropagationCascadeModel()
        dataset = self.dataset
        with open(self.cache_dir / f'{dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        sample = {'conversation_id': self.conversation_id, 'author_id': self.author_id}
        model.dataset_generator = self.dataset_generator
        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        # fig.show()

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
        sample = {'conversation_id': self.conversation_id, 'author_id': self.author_id}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=0)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        # fig.show()
        self.assertEqual(len(cascade.vs), 5)
        self.assertEqual(len(cascade.es), 0)

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
        sample = {'conversation_id': self.conversation_id, 'author_id': self.author_id}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=1)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        # fig.show()
        self.assertEqual(len(cascade.vs), 6)
        self.assertEqual(len(cascade.es), 1)

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
        sample = {'conversation_id': self.conversation_id, 'author_id': self.author_id}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=2)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        # fig.show()
        self.assertEqual(len(cascade.vs), 7)
        self.assertEqual(len(cascade.es), 2)

    def test_generate_cascade_5(self):
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
        sample = {'conversation_id': self.conversation_id, 'author_id': self.author_id}
        model.dataset_generator = self.dataset_generator
        model.pipeline = MockModel(limit=10)

        cascade = model.generate_cascade(sample)
        fig = model.plot_cascade(cascade)
        # fig.show()
        fig, ax = plt.subplots()
        ig.plot(cascade, target=ax, node_size=2)
        # plt.show()
        self.assertEqual(len(cascade.vs), 15)
        self.assertEqual(len(cascade.es), 10)

    def test_get_cascade(self):
        cascade = self.dataset_generator.get_cascade('1298573780961370112', None)
        fig, ax = plt.subplots()

        ig.plot(cascade, target=ax)
        # plt.show()

    def test_get_features_for_cascade(self):
        neighbour = self.dataset_generator.get_neighbours('2201623465')[0]
        cascade = self.dataset_generator.get_cascade(self.conversation_id, self.author_id)
        tweet_features = self.dataset_generator.get_features_for(self.conversation_id, [self.author_id], [neighbour])
        features = self.dataset_generator.generate_propagation_dataset()
        self.assertEqual(features.shape[1] - 1, tweet_features.shape[1])
        model = PropagationCascadeModel()
        with open(self.cache_dir / f'{self.dataset_generator.dataset}-model.pkl', 'rb') as f:
            model = pickle.load(f)
        model_expected_feature_columns = model.pipeline.feature_names_in_
        self.assertEqual(list(tweet_features.columns), list(model_expected_feature_columns))
        pred = model.predict(tweet_features)
