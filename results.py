import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from figures.propagation import PropagationPlotFactory
from models.propagation import PropagationModel, PropagationDatasetGenerator

logger = logging.getLogger('results')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Results:

    def __init__(self, datasets, host='localhost', port=27017, output_dir='./results', egonet_depth=2,
                 features=('propagation_tree', 'egonet', 'legitimacy_status_reputation', 'cascades', 'nodes_edges',
                           'performance'), max_cascades=None, num_samples=None):
        self.num_samples = num_samples
        self.max_cascades = max_cascades
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.datasets = datasets
        self.propagation_factory = PropagationPlotFactory(available_datasets=[self.datasets], host=self.host,
                                                          port=self.port, preload=False,
                                                          max_edges_propagation_tree=4000, )
        self.egonet_depth = egonet_depth
        self.features = features

    def plot_propagation_trees(self):
        cascades = self._get_cascades()

        for user_type, group in cascades.groupby('user_type'):
            logger.info(f'Plotting propagation trees for user type {user_type}')
            for i, row in group.iterrows():
                dataset, tweet_id = row['dataset'], row['id']
                output_dir = self.output_dir / f'propagation_trees/{user_type}'
                output_dir.mkdir(exist_ok=True, parents=True)

                try:
                    plot = self.propagation_factory.plot_propagation_tree(dataset, tweet_id)
                    plot.write_image(output_dir / f'{dataset}-{tweet_id}.png')
                    logger.info(f'Plotted propagation tree for user {tweet_id} from dataset {dataset}')
                except Exception as e:
                    logger.warning(
                        f'Failed to plot propagation tree for user {tweet_id} from dataset {dataset} due to {e}')

    def _create_cascade_pipeline(self, is_usual_suspect, party_ne=None):
        """
        Helper function to create MongoDB aggregation pipelines.
        """
        match_conditions = {
            'referenced_tweets': {'$exists': False},
            'author.remiss_metadata.is_usual_suspect': is_usual_suspect,
            'author.remiss_metadata.party': {'$ne' if party_ne else '$eq': None}
        }

        return [
            {'$match': match_conditions},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$limit': 10},
            {'$project': {'_id': 0, 'id': 1, 'retweet_count': '$public_metrics.retweet_count'}}
        ]

    def _process_data(self, raw, pipeline, schema):
        """
        Helper function to aggregate data and handle sampling.
        """
        data = raw.aggregate_pandas_all(pipeline, schema=schema)
        return data

    def _get_cascades(self):
        schema = Schema({
            'id': str,
            'retweet_count': int
        })

        client = MongoClient(self.host, self.port)

        pipelines = {
            'normal': self._create_cascade_pipeline(is_usual_suspect=False, party_ne=False),
            'suspect': self._create_cascade_pipeline(is_usual_suspect=True, party_ne=False),
            'politician': self._create_cascade_pipeline(is_usual_suspect=False, party_ne=True),
            'suspect_politician': self._create_cascade_pipeline(is_usual_suspect=True, party_ne=True)
        }

        # Initialize empty dictionaries to store data for each type
        results = {key: {} for key in pipelines.keys()}

        # Iterate over datasets and process each
        for dataset in self.datasets:
            raw = client.get_database(dataset).get_collection('raw')
            for key, pipeline in pipelines.items():
                results[key][dataset] = self._process_data(raw, pipeline, schema)

        # Concatenate results across datasets for each user type
        for key in results:
            results[key] = pd.concat(results[key].values(), keys=results[key].keys()).reset_index().drop(
                columns='level_1').rename(columns={'level_0': 'dataset'}).set_index(['dataset', 'id'])

        # Log warnings for empty dataframes
        for key in results:
            if results[key].empty:
                logger.warning(f'No {key.replace("_", " ")} users found')
            else:
                try:
                    results[key] = results[key].sample(n=10)
                except ValueError:
                    pass

        # Concatenate all user types into a single dataframe and add user_type
        cascades = pd.concat(results.values(), keys=results.keys())
        cascades = cascades.reset_index().rename(columns={'level_0': 'user_type'})
        return cascades

    def _create_users_pipeline(self, is_usual_suspect, party_ne=None):
        """
        Helper function to create MongoDB aggregation pipelines.
        """
        match_conditions = {
            'referenced_tweets': {'$exists': False},
            'author.remiss_metadata.is_usual_suspect': is_usual_suspect,
            'author.remiss_metadata.party': {'$ne' if party_ne else '$eq': None},
            'public_metrics.retweet_count': {'$gt': 100}
        }

        return [
            {'$match': match_conditions},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$group': {'_id': '$author.id', 'retweet_count': {'$sum': '$public_metrics.retweet_count'}}},
            {'$sort': {'retweet_count': -1}},
            {'$limit': 10},
            {'$project': {'_id': 0, 'author_id': '$_id', 'retweet_count': 1}}

        ]

    def _get_users(self):
        schema = Schema({
            'author_id': str,
            'retweet_count': int
        })

        client = MongoClient(self.host, self.port)

        pipelines = {
            'normal': self._create_users_pipeline(is_usual_suspect=False, party_ne=False),
            'suspect': self._create_users_pipeline(is_usual_suspect=True, party_ne=False),
            'politician': self._create_users_pipeline(is_usual_suspect=False, party_ne=True),
            'suspect_politician': self._create_users_pipeline(is_usual_suspect=True, party_ne=True)
        }

        # Initialize empty dictionaries to store data for each type
        results = {key: {} for key in pipelines.keys()}

        # Iterate over datasets and process each
        for dataset in self.datasets:
            raw = client.get_database(dataset).get_collection('raw')
            for key, pipeline in pipelines.items():
                results[key][dataset] = self._process_data(raw, pipeline, schema)

        # Concatenate results across datasets for each user type
        for key in results:
            results[key] = pd.concat(results[key].values(), keys=results[key].keys()).reset_index().drop(
                columns='level_1').rename(columns={'level_0': 'dataset'}).set_index(['dataset', 'author_id'])

        # Log warnings for empty dataframes
        for key in results:
            if results[key].empty:
                logger.warning(f'No {key.replace("_", " ")} users found')
            else:
                try:
                    results[key] = results[key].sample(n=10)
                except ValueError:
                    pass

        # Concatenate all user types into a single dataframe and add user_type
        users = pd.concat(results.values(), keys=results.keys())
        users = users.reset_index().rename(columns={'level_0': 'user_type'})
        return users

    def plot_egonet_and_backbone(self):
        logger.info('Plotting egonets and backbones')
        logger.info('Getting users')
        users = self._get_users()

        logger.info('Plotting egonets')
        # group by user type and plot the egonet for each user
        for user_type, group in users.groupby('user_type'):
            for i, row in group.iterrows():
                user_id = row['author_id']
                dataset = row['dataset']
                try:
                    fig = self.propagation_factory.plot_egonet(dataset, user_id, self.egonet_depth)
                    output_path = self.output_dir / f'egonet_and_backbone/egonet_{dataset}_{user_type}_{user_id}.png'
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    fig.write_image(output_path)
                    logger.info(
                        f'Plotted egonet for user {user_id} from dataset {dataset}')

                except Exception as e:
                    logger.warning(
                        f'Failed to plot egonet for user {user_id} from dataset {dataset} due to {e}')

    def plot_legitimacy_status_and_reputation(self):
        self.plot_legitimacy()
        # self.plot_status_and_reputation()

    def plot_legitimacy(self):
        # Get all legitimacies
        logger.info('Getting legitimacies')
        legitimacy = []
        for dataset in self.datasets:
            metadata = self.propagation_factory.get_user_metadata(dataset)
            metadata['user_type'] = metadata.apply(transform_user_type, axis=1)
            metadata['dataset'] = dataset
            legitimacy.append(metadata[['legitimacy', 'user_type', 'dataset']])

        legitimacy = pd.concat(legitimacy)
        logger.info('Plotting legitimacy distribution')
        fig = px.violin(legitimacy, y='legitimacy', color='user_type',  box=False, points='all', title='Legitimacy distribution',
                        category_orders={'user_type': ['Normal', 'Suspect', 'Politician', 'Suspect politician']})
        fig.show()
        fig.write_image(self.output_dir / 'legitimacy_status_and_reputation/legitimacy_distribution_full.png')
        legitimacy = legitimacy[legitimacy['legitimacy'] < 1000]
        fig = px.violin(legitimacy, y='legitimacy', color='user_type', box=False, points='all', title='Legitimacy distribution',
                        category_orders={'user_type': ['Normal', 'Suspect', 'Politician', 'Suspect politician']})
        fig.show()
        fig.write_image(self.output_dir / 'legitimacy_status_and_reputation/legitimacy_distribution_1000.png')



    def plot_status_and_reputation(self):
        logger.info('Getting users')
        # get degrees and metadata
        users = self._get_users()
        # group by user type and plot the egonet for each user
        for user_type, group in users.groupby('user_type'):
            for i, row in group.iterrows():
                dataset = row['dataset']
                author_id = row['author_id']
                try:
                    reputation = self.propagation_factory.network_metrics.load_reputation_for_author(dataset, author_id)
                    status = self.propagation_factory.network_metrics.load_status_for_author(dataset, author_id)
                    output_dir = self.output_dir / 'legitimacy_status_and_reputation' / 'reputation'
                    output_dir.mkdir(exist_ok=True, parents=True)
                    fig = self._plot_reputation(reputation)
                    fig.write_image(f'{output_dir}/{user_type}_{dataset}_{author_id}.png')
                    fig = self._plot_status(status)
                    output_dir = self.output_dir / 'legitimacy_status_and_reputation' / 'status'
                    output_dir.mkdir(exist_ok=True, parents=True)
                    fig.write_image(f'{output_dir}/{user_type}_{dataset}_{author_id}.png')

                    logger.info(f'Plotted reputation and status for '
                                f'{author_id} from dataset {dataset} of {user_type}')

                except Exception as e:
                    logger.warning(
                        f'Failed to plot legitimacy, reputation and status for user  from dataset {dataset} due to {e}')

    def _plot_reputation(self, reputation):
        return self._plot_time_series(reputation, 'Reputation over time', 'Time', 'Reputation')

    def _plot_status(self, status):
        return self._plot_time_series(status, 'Status over time', 'Time', 'Status')

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
        for dataset in self.datasets:
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
        output_path = self.output_dir / 'nodes_and_edges/nodes_edges.csv'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        num_nodes_edges.to_csv(output_path)

    def generate_performance_table(self):
        logger.info('Generating performance table')
        results_filepath = self.output_dir / 'performance/model_performance.csv'
        results_filepath.parent.mkdir(exist_ok=True, parents=True)

        results = {}
        for dataset in self.datasets:
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
        results.to_markdown(self.output_dir / 'performance/model_performance.md')

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
        model = PropagationModel()
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
        train_confmat.to_csv(self.output_dir / f'performance/{dataset}' / 'train_confmat.csv')
        test_confmat = pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted'])
        test_confmat.to_csv(self.output_dir / f'performance/{dataset}' / 'test_confmat.csv')
        # plot feature importance
        feature_importances = pd.Series(model.model['classifier'].feature_importances_,
                                        index=model.model['classifier'].feature_names_in_).sort_values(ascending=False)
        # take top 20 features
        feature_importances = feature_importances.head(20)
        fig = px.bar(feature_importances, title='Feature importance')
        fig.update_xaxes(title_text='Feature')
        fig.update_yaxes(title_text='Importance')
        fig.write_image(self.output_dir / f'performance/{dataset}' / 'feature_importance.png')
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


def run_results(datasets=('Openarms', 'MENA_Agressions', 'MENA_Ajudes', 'Barcelona_2019', 'Generales_2019',
                          'Generalitat_2021', 'Andalucia_2022'),
                host='localhost', port=27017, output_dir='./results', egonet_depth=2,
                features=('propagation_tree', 'egonet', 'legitimacy', 'cascades', 'nodes_edges', 'performance'),
                max_cascades=None, num_samples=None):
    logger.info('Running results')
    logger.info(f'Datasets: {datasets}')
    logger.info(f'Host: {host}')
    logger.info(f'Port: {port}')
    logger.info(f'Output directory: {output_dir}')
    logger.info(f'Egonet depth: {egonet_depth}')
    logger.info(f'Max cascades: {max_cascades}')
    logger.info(f'Num samples: {num_samples}')
    logger.info(f'Features: {features}')
    results = Results(datasets=datasets, host=host, port=port, output_dir=output_dir,
                      egonet_depth=egonet_depth,
                      features=features,
                      max_cascades=max_cascades,
                      num_samples=num_samples)
    results.process()


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'

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
    # run_results(['test_dataset_2', 'test_dataset_3'],
    #             host='localhost',
    #             features=('legitimacy_status_reputation',),
    #             output_dir='results/local/',
    #             num_samples=1000)
    # run_results(['Openarms', 'MENA_Agressions', 'MENA_Ajudes'],
    #             host='mongodb://srvinv02.esade.es',
    #             features=('performance',),
    #             output_dir='results/performance',
    #             num_samples=50000)
    run_results(
        host='mongodb://srvinv02.esade.es',
        features=['nodes_edges'],
        output_dir='results/final',
        num_samples=50000
    )
