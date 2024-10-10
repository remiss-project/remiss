import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from figures.propagation import PropagationPlotFactory
from models.propagation import PropagationCascadeModel, PropagationDatasetGenerator

logger = logging.getLogger('results')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Results:

    def __init__(self, dataset, host='localhost', port=27017, output_dir='./results', egonet_depth=2,
                 features=('propagation_tree', 'egonet', 'legitimacy_status_reputation', 'cascades', 'nodes_edges',
                           'performance'), max_cascades=None, num_samples=None):
        self.num_samples = num_samples
        self.max_cascades = max_cascades
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.dataset = dataset
        self.propagation_factory = PropagationPlotFactory(available_datasets=[self.dataset], host=self.host,
                                                          port=self.port, preload=False)
        self.egonet_depth = egonet_depth
        self.features = features

    def plot_propagation_trees(self):
        normal, suspect, politician, suspect_politician = self._get_cascades()
        for i, row in normal.iterrows():
            try:
                plot = self.propagation_factory.plot_propagation_tree(self.dataset, row['id'])
                plot.write_image(f'{self.output_dir}/normal_propagation_tree_{i}.png')
                logger.info(f'Plotted propagation tree for user {row["id"]} from dataset {self.dataset}')
            except Exception as e:
                logger.warning(
                    f'Failed to plot propagation tree for user {row["id"]} from dataset {self.dataset} due to {e}')

        for i, row in suspect.iterrows():
            try:
                plot = self.propagation_factory.plot_propagation_tree(self.dataset, row['id'])
                plot.write_image(f'{self.output_dir}/suspect_propagation_tree_{i}.png')
                logger.info(f'Plotted propagation tree for user {row["id"]} from dataset {self.dataset}')
            except Exception as e:
                logger.warning(
                    f'Failed to plot propagation tree for user {row["id"]} from dataset {self.dataset} due to {e}')

        for i, row in politician.iterrows():
            try:
                plot = self.propagation_factory.plot_propagation_tree(self.dataset, row['id'])
                plot.write_image(f'{self.output_dir}/politician_propagation_tree_{i}.png')
                logger.info(f'Plotted propagation tree for user {row["id"]} from dataset {self.dataset}')
            except Exception as e:
                logger.warning(
                    f'Failed to plot propagation tree for user {row["id"]} from dataset {self.dataset} due to {e}')

        for i, row in suspect_politician.iterrows():
            try:
                plot = self.propagation_factory.plot_propagation_tree(self.dataset, row['id'])
                plot.write_image(f'{self.output_dir}/suspect_politician_propagation_tree_{i}.png')
                logger.info(f'Plotted propagation tree for user {row["id"]} from dataset {self.dataset}')
            except Exception as e:
                logger.warning(
                    f'Failed to plot propagation tree for user {row["id"]} from dataset {self.dataset} due to {e}')

    def _get_cascades(self):
        normal_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False},
                        'author.remiss_metadata.is_usual_suspect': False,
                        'author.remiss_metadata.party': {'$eq': None},
                        }},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$limit': 100},
            {'$project': {'_id': 0, 'id': 1, 'retweet_count': '$public_metrics.retweet_count'}}
        ]
        suspect_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False},
                        'author.remiss_metadata.is_usual_suspect': True,
                        'author.remiss_metadata.party': {'$eq': None},
                        }},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$limit': 100},
            {'$project': {'_id': 0, 'id': 1, 'retweet_count': '$public_metrics.retweet_count'}}
        ]
        politician_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False},
                        'author.remiss_metadata.is_usual_suspect': False,
                        'author.remiss_metadata.party': {'$ne': None},
                        }},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$limit': 100},
            {'$project': {'_id': 0, 'id': 1, 'retweet_count': '$public_metrics.retweet_count'}}
        ]
        suspect_politician_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False},
                        'author.remiss_metadata.is_usual_suspect': True,
                        'author.remiss_metadata.party': {'$ne': None},
                        }},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$limit': 100},
            {'$project': {'_id': 0, 'id': 1, 'retweet_count': '$public_metrics.retweet_count'}}
        ]

        client = MongoClient(self.host, self.port)
        raw = client.get_database(self.dataset).get_collection('raw')
        normal = raw.aggregate_pandas_all(normal_pipeline)
        suspect = raw.aggregate_pandas_all(suspect_pipeline)
        politician = raw.aggregate_pandas_all(politician_pipeline)
        suspect_politician = raw.aggregate_pandas_all(suspect_politician_pipeline)
        if normal.empty:
            logger.warning('No normal users found')
        if suspect.empty:
            logger.warning('No suspect users found')
        if politician.empty:
            logger.warning('No politician users found')
        if suspect_politician.empty:
            logger.warning('No suspect politician users found')
        return normal, suspect, politician, suspect_politician

    def plot_egonet_and_backbone(self):
        logger.info('Plotting egonets and backbones')
        logger.info('Getting degrees')
        degrees = self._get_degrees()

        logger.info('Plotting egonets')
        # group by user type and plot the egonet for each user
        for user_type, group in degrees.groupby('user_type'):
            figures = []

            for i, row in group.iterrows():
                try:
                    plot = self.propagation_factory.plot_egonet(self.dataset, row['index'], self.egonet_depth)
                    figures.append(plot)
                    logger.info(
                        f'Plotted egonet for user {row["index"]} from dataset {self.dataset}')
                    break
                except Exception as e:
                    logger.warning(
                        f'Failed to plot egonet for user {row["index"]} from dataset {self.dataset} due to {e}')

            # save the plotly figures as png
            for i, fig in enumerate(figures):
                fig.write_image(f'{self.output_dir}/egonet_{user_type}_{i}.png')

    def _get_degrees(self):

        hidden_network = self.propagation_factory.egonet.get_hidden_network(self.dataset)
        degree = pd.Series(hidden_network.degree(), index=hidden_network.vs['author_id'])
        metadata = self._get_metadata(self.dataset)
        # merge degree with metadata
        degree = degree.to_frame('degree').join(metadata.set_index('author_id'))

        # merge dfs with the dataset as a additional column
        degree = degree.dropna(subset='is_usual_suspect').sort_values(ascending=False, by='degree').reset_index()
        return degree

    def _get_metadata(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # get a table with author_id, party, is_usual_suspect
        pipeline = [
            {'$project': {'_id': 0, 'author_id': 1, 'party': '$author.remiss_metadata.party',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect'}},
        ]
        schema = Schema({'author_id': str, 'party': str, 'is_usual_suspect': bool})

        df = collection.aggregate_pandas_all(pipeline, schema=schema)

        def get_user_type(row):
            if row['is_usual_suspect']:
                if row['party']:
                    return 'suspect_politician'
                return 'suspect'
            if row['party']:
                return 'politician'
            return 'normal'

        df['user_type'] = df.apply(get_user_type, axis=1)
        return df

    def plot_legitimacy_status_and_reputation(self):
        logger.info('Plotting legitimacy, status and reputation')
        logger.info('Getting legitimacy, status and reputation')
        legitimacy = self.propagation_factory.network_metrics.get_legitimacy(self.dataset)
        reputation = self.propagation_factory.network_metrics.get_reputation(self.dataset)
        status = self.propagation_factory.network_metrics.get_status(self.dataset)

        logger.info('Getting degrees')
        # get degrees and metadata
        degrees = self._get_degrees()
        legitimacy_figures_data = defaultdict(list)
        # group by user type and plot the egonet for each user
        for user_type, group in degrees.groupby('user_type'):
            reputation_figures = []
            status_figures = []
            for i, row in group.iterrows():
                try:
                    user_reputation = reputation.loc[row['index']]
                    user_status = status.loc[row['index']]
                    fig = self._plot_reputation(user_reputation)
                    reputation_figures.append(fig)
                    fig = self._plot_status(user_status)
                    status_figures.append(fig)
                    legitimacy_figures_data[user_type].append(legitimacy[row['index']])

                    logger.info(f'Plotted legitimacy, reputation and status for '
                                f'user from dataset {self.dataset}')
                    break
                except Exception as e:
                    logger.warning(
                        f'Failed to plot legitimacy, reputation and status for user  from dataset {self.dataset} due to {e}')

            # save the plotly figures as png
            for i, fig in enumerate(reputation_figures):
                fig.write_image(f'{self.output_dir}/reputation_{user_type}_{i}.png')
            for i, fig in enumerate(status_figures):
                fig.write_image(f'{self.output_dir}/status_{user_type}_{i}.png')

        legitimacy_figures = self._plot_legitimacy(legitimacy_figures_data)
        for i, fig in enumerate(legitimacy_figures):
            fig.write_image(f'{self.output_dir}/legitimacy_{i}.png')

    def _plot_reputation(self, reputation):
        return self._plot_time_series(reputation, 'Reputation over time', 'Time', 'Reputation')

    def _plot_status(self, status):
        return self._plot_time_series(status, 'Status over time', 'Time', 'Status')

    def _plot_legitimacy(self, data):
        data = pd.DataFrame(data)
        figures = []
        for i, row in data.iterrows():
            fig = px.bar(row, x=row.index, y=row.values, color=row.index)
            fig.update_layout(title='Legitimacy', xaxis_title='User type', yaxis_title='Legitimacy')
            figures.append(fig)
        return figures

    def _plot_time_series(self, data, title, x_label, y_label):
        # Assume data is a pd.Series
        data.name = y_label
        fig = px.line(data)
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text=y_label)
        # fig.update_layout(title=title)
        return fig

    def generate_nodes_and_edges_table(self, max_nodes=80000):
        logger.info('Generating nodes and edges table')
        num_nodes_edges = {}
        for dataset in self.dataset:
            logger.info(f'Getting hidden network for dataset {dataset}')
            # hidden_network = self.propagation_factory.egonet.load_hidden_network_backbone(dataset)
            hidden_network = self.propagation_factory.egonet.load_hidden_network(dataset)
            num_nodes = hidden_network.vcount()
            num_edges = hidden_network.ecount()
            start_time = datetime.now()
            logger.info('Computing closeness')
            if max_nodes and hidden_network.vcount() > max_nodes:
                hidden_network = hidden_network.subgraph(np.random.choice(hidden_network.vs, max_nodes, replace=False))
            closeness = hidden_network.closeness()
            closeness = pd.Series(closeness).mean()
            logger.info(f'Closeness computed in {datetime.now() - start_time}')
            num_nodes_edges[dataset] = {'num_nodes': num_nodes, 'num_edges': num_edges,
                                        'closeness': closeness}

        num_nodes_edges = pd.DataFrame(num_nodes_edges).T
        num_nodes_edges.to_csv(self.output_dir / 'nodes_edges.csv')

    def generate_performance_table(self):
        logger.info('Generating performance table')
        results_filepath = self.output_dir / 'model_performance.csv'

        results = {}
        for dataset in self.dataset:
            try:
                logger.info(f'Testing dataset {dataset}')
                X, y = self._load_dataset(self.output_dir, dataset)
                results[dataset] = self._test_dataset(X, y, dataset)

            except Exception as e:
                logger.error(f'Failed to test dataset {dataset} due to {e}')

        results = pd.DataFrame(results).T
        logger.info('Results')
        pd.set_option('display.max_columns', None)
        logger.info(results)
        logger.info(f'Saving results to {results_filepath}')
        results.to_csv(results_filepath, index=True)
        results.to_markdown(self.output_dir / 'model_performance.md')

    def _load_dataset(self, output_dir, dataset):
        dataset_path = Path(output_dir) / f'{dataset}.csv'
        if dataset_path.exists():
            logger.info(f'Loading dataset {dataset} from {dataset_path}')
            features = pd.read_csv(dataset_path)
        else:
            logger.info(f'Dataset {dataset} not found. Generating dataset')
            dataset_generator = PropagationDatasetGenerator(dataset, host=self.host, port=self.port,
                                                            num_samples=self.num_samples)
            features = dataset_generator.generate_propagation_dataset()
            features.to_csv(dataset_path, index=False)
        y = features['propagated']
        X = features.drop(columns=['propagated'])
        return X, y


    def _test_dataset(self, X, y, dataset):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = PropagationCascadeModel()
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = self._get_metrics(y_train, y_train_pred)
        test_metrics = self._get_metrics(y_test, y_test_pred)

        results = {
            'train': train_metrics,
            'test': test_metrics
        }
        results = pd.DataFrame(results).stack().swaplevel().sort_index(ascending=False)
        results.name = dataset


        train_confmat = pd.crosstab(y_train, y_train_pred, rownames=['Actual'], colnames=['Predicted'])
        (self.output_dir / f'{dataset}').mkdir(exist_ok=True, parents=True)
        train_confmat.to_csv(self.output_dir / f'{dataset}' / 'train_confmat.csv')
        test_confmat = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])
        test_confmat.to_csv(self.output_dir / f'{dataset}' / 'test_confmat.csv')
        # plot feature importance
        feature_importances = pd.Series(model.model['classifier'].feature_importances_,
                                        index=model.model['classifier'].feature_names_in_).sort_values(ascending=False)
        # take top 20 features
        feature_importances = feature_importances.head(20)
        fig = px.bar(feature_importances, title='Feature importance')
        fig.update_xaxes(title_text='Feature')
        fig.update_yaxes(title_text='Importance')
        fig.write_image(self.output_dir / f'{dataset}' / 'feature_importance.png')
        return results

    def _get_metrics(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {'accuracy': report['accuracy'],
                   'precision': report['macro avg']['precision'],
                   'recall': report['macro avg']['recall'],
                   'f1_score': report['macro avg']['f1-score'],
                   'No Propagation support': report['0']['support'],
                   'Propagation support': report['1']['support']
                   }

        return metrics

    def process(self):
        logger.info('Processing results')
        for feature in self.features:
            match feature:
                case 'propagation_trees':
                    self.plot_propagation_trees()
                case 'egonet':
                    self.plot_egonet_and_backbone()
                case 'legitimacy_status_reputation':
                    self.plot_legitimacy_status_and_reputation()

                case 'nodes_edges':
                    self.generate_nodes_and_edges_table()
                case 'performance':
                    self.generate_performance_table()
                case _:
                    logger.error(f'Feature {feature} not recognized')

        logger.info('Results processed')


def run_results(dataset, host='localhost', port=27017, output_dir='./results', egonet_depth=2,
                features=('propagation_tree', 'egonet', 'legitimacy', 'cascades', 'nodes_edges', 'performance'),
                max_cascades=None, num_samples=None):
    logger.info('Running results')
    logger.info(f'Datasets: {dataset}')
    logger.info(f'Host: {host}')
    logger.info(f'Port: {port}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Egonet depth: {egonet_depth}')
    logger.info(f'Max cascades: {max_cascades}')
    logger.info(f'Num samples: {num_samples}')
    logger.info(f'Features: {features}')
    results = Results(dataset=dataset, host=host, port=port, output_dir=output_dir,
                      egonet_depth=egonet_depth,
                      features=features,
                      max_cascades=max_cascades,
                      num_samples=num_samples)
    results.process()


if __name__ == '__main__':
    # fire.Fire(run_results)
    #  - Openarms
    # - MENA_Agressions
    # - MENA_Ajudes
    # - Barcelona_2019
    # - Generales_2019
    # - Generalitat_2021
    # - Andalucia_2022
    # run_results(['Openarms', 'MENA_Agressions', 'MENA_Ajudes', 'Barcelona_2019', 'Generales_2019', 'Generalitat_2021',
    #              'Andalucia_2022'],
    #             host='mongodb://srvinv02.esade.es',
    #             features=['nodes_edges'],
    #             output_dir='results/nodes_edges')
    # run_results('Openarms',
    #             host='mongodb://srvinv02.esade.es',
    #             features=('legitimacy_status_reputation',),
    #             output_dir='results/openarms')
    # run_results(['test_dataset_2'],
    #             host='localhost',
    #             features=('performance',),
    #             output_dir='results/local/performance',
    #             num_samples=1000)
    # run_results(['Openarms', 'MENA_Agressions', 'MENA_Ajudes'],
    #             host='mongodb://srvinv02.esade.es',
    #             features=('performance',),
    #             output_dir='results/performance',
    #             num_samples=50000)
    run_results(['Andalucia_2022', 'Barcelona_2019', 'Generales_2019', 'Generalitat_2021'],
                host='localhost',
                features=['performance'],
                output_dir='final_results/performance',
                num_samples=50000
        )

