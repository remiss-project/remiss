import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd

from models.propagation import PropagationDatasetGenerator, PropagationModel, CascadeGenerator


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
        self.assertEqual(len(user_features), 22)

    def test_prepare_propagation_dataset(self):
        dataset = self.dataset
        features = self.dataset_generator.generate_propagation_dataset()

        features.to_csv(self.cache_dir / f'{dataset}-features.csv')
        # num_samples = np.minimum(100, features.shape[0])
        # sns.pairplot(features.sample(num_samples), hue='propagated', diag_kind='kde')
        # plt.savefig('tmp/cache_propagation_2/pairplot.png')

    def test_fit(self):
        model = PropagationModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        features = features.sample(frac=0.01)
        X, y = features.drop(columns=['propagated']), features['propagated']
        model.fit(X, y)


    def test_predict(self):
        model = PropagationModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        features = features.sample(frac=0.01)

        X, y = features.drop(columns=['propagated']), features['propagated']
        model.fit(X, y)

        y_pred = model.predict(X)
        print(y_pred)

    def test_predict_proba(self):
        model = PropagationModel()
        dataset = self.dataset
        features = pd.read_csv(self.cache_dir / f'{dataset}-features.csv', index_col=0)
        features = features.sample(frac=0.01)

        X, y = features.drop(columns=['propagated']), features['propagated']
        model.fit(X, y)

        y_pred = model.predict_proba(X)
        print(y_pred)

    def test_generate_cascade(self):
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
        model = MockModel(limit=5)

        cascade_generator = CascadeGenerator(model=model, dataset_generator=self.dataset_generator)
        cascade = cascade_generator.generate_cascade(self.test_tweet_id)
        self.dataset_generator.diffusion_metrics._plot_graph_igraph(cascade)