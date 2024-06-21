import random
import unittest
from pathlib import Path
import plotly.express as px

import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Timestamp
from pymongo import MongoClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

from figures.propagation import PropagationPlotFactory
from tests.conftest import populate_test_database, delete_test_database
import igraph as ig
import seaborn as sns


class PropagationTestCase(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     populate_test_database('test_dataset')
    #     populate_test_database('test_dataset_small', small=True)

    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database(self.dataset)

    def setUp(self):
        self.plot_factory = PropagationPlotFactory()
        self.dataset = 'test_dataset'
        self.dataset_small = 'test_dataset_small'

    def test_propagation_tree(self):
        graph = self.plot_factory.get_propagation_tree(self.dataset, '1160842257647493120')
        self.assertEqual(graph.vcount(), 76)
        self.assertEqual(graph.ecount(), 75)
        self.assertEqual(graph.is_directed(), True)
        self.assertIsInstance(graph.vs['author_id'][0], str)
        self.assertIsInstance(graph.vs['username'][0], str)
        self.assertIsInstance(graph.vs['tweet_id'][0], str)
        self.assertEqual(graph.vs['party'][0], None)
        self.assertEqual(graph.vs['is_usual_suspect'][0], False)
        self.assertIsInstance(graph.vs['created_at'][0], Timestamp)

        # Display the igraph graph with matplotlib
        layout = graph.layout(self.plot_factory.layout)
        fig, ax = plt.subplots()
        ig.plot(graph, layout=layout, target=ax)
        plt.show()

    def test_all_connected_to_conversation_id(self):
        graph = self.plot_factory.get_propagation_tree(self.dataset, '1160842257647493120')
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)
        shortest_paths = self.plot_factory.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_propagation_lengths(self):
        df = self.plot_factory.get_conversation_sizes(self.dataset)
        for tweet_id in df['conversation_id'].iloc[11:]:
            try:
                graph = self.plot_factory.get_propagation_tree(self.dataset, tweet_id)
                layout = graph.layout(self.plot_factory.layout)
                fig, ax = plt.subplots()
                ax.set_title(f'Conversation {tweet_id}')
                ig.plot(graph, layout=layout, target=ax)
                plt.show()
                print(f'Conversation {tweet_id} works')
                break
            except (RuntimeError, KeyError) as ex:
                pass
        print(df)
        df.hist(log=True, bins=100)
        plt.show()

    def test_tweets_with_references(self):
        conversation_id, tweets, references = self.plot_factory.get_conversation(self.dataset, '1160842257647493120')
        self.assertEqual(tweets.columns.tolist(),
                         ['tweet_id', 'author_id', 'username', 'is_usual_suspect', 'party', 'created_at'])
        self.assertEqual(len(references), 61)
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
        timestamps = [Timestamp.now() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    'remiss_metadata': {
                        'is_usual_suspect': random.choice([True, False]),
                        'party': random.choice([None, 'party1', 'party2']),
                    }
                    } for i in range(8)]
        collection.insert_many([{
            'id': str(i),
            'conversation_id': '0',
            'referenced_tweets': [{'type': 'replied_to',
                                   'id': str(j),
                                   'author': authors[j],
                                   'created_at': timestamps[j]}],
            'username': f'username_{i}',
            'author': authors[i],
            'created_at': timestamps[i]

        } for i, j in edges])
        graph = self.plot_factory.get_propagation_tree(dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set((graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es)
        expected_edges = {(f'username_{source}', f'username_{target}') for source, target in edges}
        self.assertEqual(actual_edges, expected_edges)

    def test_propagation_tree_disconnected(self):
        client = MongoClient('localhost', 27017)
        dataset = 'test_simple_dataset'
        client.drop_database(dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        edges = [(0, 1), (1, 2), (2, 4), (2, 5), (3, 6), (3, 7)]
        original_graph = ig.Graph(n=8, edges=edges, directed=True)
        original_graph.vs['label'] = [str(i) for i in range(8)]
        fig, ax = plt.subplots()
        layout = original_graph.layout(self.plot_factory.layout)
        ig.plot(original_graph, layout=layout, target=ax)
        timestamps = [Timestamp.now().date() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    'remiss_metadata': {
                        'is_usual_suspect': random.choice([True, False]),
                        'party': random.choice([None, 'party1', 'party2']),
                    }
                    } for i in range(8)]
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        test_data = []
        for source, targets in edges_df.groupby('source'):
            test_data.append({
                'id': str(source),
                'conversation_id': '0',
                'referenced_tweets': [{'type': 'replied_to',
                                       'id': str(target),
                                       'author': authors[target],
                                       'created_at': timestamps[target]} for target in targets['target']],
                'username': f'username_{source}',
                'author': authors[source],
                'created_at': timestamps[source]
            })
        collection.insert_many(test_data)
        references_created_at = []
        for tweet in test_data:
            for reference in tweet['referenced_tweets']:
                references_created_at.append((reference['id'], reference['created_at']))

        references_created_at = pd.DataFrame(references_created_at, columns=['id', 'created_at'])

        graph = self.plot_factory.get_propagation_tree(dataset, '0')
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = set((graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es)
        expected_edges = {(f'username_{source}', f'username_{target}') for source, target in edges}
        expected_edges.add(('username_0', 'username_3'))
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)

        shortest_paths = self.plot_factory.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_propagation_tree_missing_conversation_id(self):
        client = MongoClient('localhost', 27017)
        dataset = 'test_simple_dataset'
        client.drop_database(dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        edges = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        original_graph = ig.Graph(n=8, edges=edges, directed=True)
        original_graph.vs['label'] = [str(i) for i in range(8)]
        fig, ax = plt.subplots()
        layout = original_graph.layout(self.plot_factory.layout)
        ig.plot(original_graph, layout=layout, target=ax)
        timestamps = [Timestamp.now().date() + pd.offsets.Hour(i) for i in range(8)]
        authors = [{'id': f'author_id_{i}',
                    'username': f'username_{i}',
                    'remiss_metadata': {
                        'is_usual_suspect': random.choice([True, False]),
                        'party': random.choice([None, 'party1', 'party2']),
                    }
                    } for i in range(8)]
        edges_df = pd.DataFrame(edges, columns=['source', 'target'])
        test_data = []
        for source, targets in edges_df.groupby('source'):
            test_data.append({
                'id': str(source),
                'conversation_id': '0',
                'referenced_tweets': [{'type': 'replied_to',
                                       'id': str(target),
                                       'author': authors[target],
                                       'created_at': timestamps[target]} for target in targets['target']],
                'username': f'username_{source}',
                'author': authors[source],
                'created_at': timestamps[source]
            })
        collection.insert_many(test_data)
        references_created_at = []
        for tweet in test_data:
            for reference in tweet['referenced_tweets']:
                references_created_at.append((reference['id'], reference['created_at']))

        references_created_at = pd.DataFrame(references_created_at, columns=['id', 'created_at'])

        graph = self.plot_factory.get_propagation_tree(dataset, '1')
        fig, ax = plt.subplots()
        layout = graph.layout(self.plot_factory.layout)
        ig.plot(graph, layout=layout, target=ax)
        plt.show()
        actual_edges = pd.DataFrame(
            [(graph.vs['label'][edge.source], graph.vs['label'][edge.target]) for edge in graph.es],
            columns=['source', 'target'])
        actual_edges['source'] = actual_edges['source'].str.replace('-', 'username_0').astype(str)
        actual_edges['target'] = actual_edges['target'].str.replace('-', 'username_0').astype(str)
        actual_edges = set(actual_edges.itertuples(index=False, name=None))
        edges.append((0, 1))
        expected_edges = {(f'username_{source}', f'username_{target}') for source, target in edges}
        self.assertEqual(actual_edges, expected_edges)
        self.assertEqual(len(graph.connected_components(mode='weak')), 1)

        shortest_paths = self.plot_factory.get_shortest_paths_to_conversation_id(graph)
        self.assertFalse(shortest_paths.isna().any())

    def test_propagation_tree_plot(self):
        graph = self.plot_factory.get_propagation_tree(self.dataset, '1160842257647493120')
        fig = self.plot_factory.get_propagation_figure(graph)
        fig.show()

    def test_conversation_no_nat(self):
        conversation_id, tweets, references = self.plot_factory.get_conversation(self.dataset, '1160842257647493120')
        tweets = tweets.drop(columns=['party'])
        self.assertFalse(tweets.isna().any().any())
        self.assertFalse(references.isna().any().any())

    def test_depth_plot(self):
        fig = self.plot_factory.plot_depth_over_time(self.dataset, '1160842257647493120')
        fig.show()

    def test_size_plot(self):
        fig = self.plot_factory.plot_size_over_time(self.dataset, '1160842257647493120')
        fig.show()

    def test_max_breadth_plot(self):
        fig = self.plot_factory.plot_max_breadth_over_time(self.dataset, '1160842257647493120')
        fig.show()

    def test_structured_virality_plot(self):
        fig = self.plot_factory.plot_structural_virality_over_time(self.dataset, '1160842257647493120')
        fig.show()

    def test_plot_propagation_tree(self):
        fig = self.plot_factory.plot_propagation_tree(self.dataset, '1160842257647493120')
        fig.show()

    def test_graph_propagation(self):
        graph = self.plot_factory.get_full_graph(self.dataset)
        components = graph.connected_components(mode='weak')
        self.assertEqual(len(components), 8485)

    def test_depth_cascade_ccdf_plot(self):
        start_time = Timestamp.now()
        fig = self.plot_factory.plot_depth_cascade_ccdf(self.dataset_small)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        fig.show()

    def test_size_cascade_ccdf_plot(self):
        start_time = Timestamp.now()
        fig = self.plot_factory.plot_size_cascade_ccdf(self.dataset)
        end_time = Timestamp.now()
        print(f'Time taken: {end_time - start_time}')
        fig.show()

    def test_cascade_count_over_time_plot(self):
        fig = self.plot_factory.plot_cascade_count_over_time(self.dataset)
        fig.show()

    def test_persist_propagation_metrics(self):
        self.plot_factory.persist_propagation_metrics(self.dataset_small)
        expected = self.plot_factory.get_structural_viralities(self.dataset_small)
        actual = self.plot_factory.load_propagation_metrics_from_db(self.dataset_small)
        pd.testing.assert_frame_equal(expected, actual)

    def test_prepare_propagation_dataset(self):
        dataset = self.dataset
        self.plot_factory.cache_dir = Path('tmp/cache_propagation_2')
        features = self.plot_factory.generate_propagation_dataset(dataset)
        features.to_csv(self.plot_factory.cache_dir / 'features.csv')
        features['propagated'].hist()
        # plot heatmap with seaborn
        # sns.pairplot(features, hue='propagated', diag_kind='kde')

        plt.show()

    def test_features_fail(self):
        features = pd.read_csv('tmp/cache_propagation_2/features.csv', index_col=0)
        sample = features.sample(10000)
        sns.pairplot(sample, hue='propagated', diag_kind='kde')
        plt.savefig('tmp/cache_propagation_2/pairplot.png')

        plt.show()

    def test_fit_fail(self):
        df = pd.read_csv('tmp/cache_propagation_2/features.csv', index_col=0)

        X, y = df.drop(columns='propagated'), df['propagated']
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('transformer', ColumnTransformer([
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 X.select_dtypes(include='object').columns),
            ], remainder='passthrough')),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier())
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        # show plotly histogram for y_train
        fig = px.histogram(y_train, title='Distribution of labels in the training set')
        fig.update_xaxes(title_text='Label')
        fig.update_yaxes(title_text='Count')
        fig.show()
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        print('Training set metrics')
        print(classification_report(y_train, y_train_pred))
        print('Test set metrics')
        print(classification_report(y_test, y_test_pred))
        # plot feature importance
        feature_importances = pd.Series(pipeline['classifier'].feature_importances_,
                                        index=pipeline['classifier'].feature_names_in_).sort_values(ascending=False)
        fig = px.bar(feature_importances, title='Feature importance')
        fig.update_xaxes(title_text='Feature')
        fig.update_yaxes(title_text='Importance')
        fig.show()

    def test_fit_propagation_model(self):
        dataset = self.dataset
        model = self.plot_factory.fit_propagation_model(dataset)
        self.assertIsNotNone(model)

    def test_conversation_ids(self):
        conversation_ids = self.plot_factory.get_conversation_ids(self.dataset_small)
        # All the conversation ids should exist in the database, thus retrieving their graph should not raise an error
        for conversation_id in tqdm(conversation_ids['conversation_id']):
            try:
                self.plot_factory.get_propagation_tree(self.dataset_small, conversation_id)
            except Exception as ex:
                self.fail(f'Conversation id {conversation_id} raised an exception: {ex}')


if __name__ == '__main__':
    unittest.main()
