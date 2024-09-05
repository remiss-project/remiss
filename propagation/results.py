import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from figures.propagation import PropagationPlotFactory
from models.propagation import PropagationCascadeModel, PropagationDatasetGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


class Results:

    def __init__(self, datasets, host='localhost', port=27017, output_dir='./results', top_n=10, egonet_depth=2):
        self.top_n = top_n
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.datasets = datasets
        self.propagation_factory = PropagationPlotFactory(available_datasets=self.datasets, host=self.host,
                                                          port=self.port)
        self.egonet_depth = egonet_depth

    def plot_propagation_tree(self):
        logger.info('Plotting propagation trees')
        logger.info('Getting conversations')
        conversations = self._get_conversations()

        # Since some  might fail because the original conversation id tweet is not present, just keep sampling
        # until we have the top_n
        logger.info('Plotting propagation trees')
        count = 0
        figures = []
        for i, (dataset, conversation_id) in conversations[['dataset', 'conversation_id']].iterrows():
            if count == self.top_n:
                break
            try:
                plot = self.propagation_factory.plot_propagation_tree(dataset, conversation_id)
                figures.append(plot)

                count += 1
                logger.info(f'Plotted conversation {conversation_id} from dataset {dataset} count {count}')
            except Exception as e:
                logger.warning(f'Failed to plot conversation {conversation_id} from dataset {dataset} due to {e}')

        # save the plotly figures as png
        for i, fig in enumerate(figures):
            fig.write_image(f'{self.output_dir}/propagation_tree_{i}.png')

    def _get_conversations(self):
        conversation_sizes = {}
        for dataset in self.datasets:
            conversation_sizes[dataset] = self.propagation_factory.diffusion_metrics.get_conversation_sizes(dataset)

        # merge dfs with the dataset as a additional column
        conversation_sizes = pd.concat(conversation_sizes, names=['dataset'])
        # Sort by size
        conversations = conversation_sizes.sort_values(ascending=False, by='size').reset_index()

        def get_user_type(row):
            if row['is_usual_suspect']:
                if row['party']:
                    return 'suspect_politician'
                return 'suspect'
            if row['party']:
                return 'politician'
            return 'normal'

        conversations['user_type'] = conversations.apply(get_user_type, axis=1)
        return conversations

    def plot_egonet_and_backbone(self):
        logger.info('Plotting egonets and backbones')
        logger.info('Getting degrees')
        degrees = self._get_degrees()

        logger.info('Plotting egonets')
        # group by user type and plot the egonet for each user
        for user_type, group in degrees.groupby('user_type'):
            count = 0
            figures = []

            for i, row in group.iterrows():
                if count == self.top_n:
                    break
                try:
                    plot = self.propagation_factory.plot_egonet(row['dataset'], row['author_id'], self.egonet_depth)
                    figures.append(plot)
                    count += 1
                    logger.info(
                        f'Plotted egonet for user {row["author_id"]} from dataset {row["dataset"]} count {count}')
                except Exception as e:
                    logger.warning(f'Failed to plot egonet for user {row["author_id"]} from dataset {row["dataset"]} due to {e}')

            # save the plotly figures as png
            for i, fig in enumerate(figures):
                fig.write_image(f'{self.output_dir}/egonet_{user_type}_{i}.png')

    def _get_degrees(self):
        degrees = {}
        for dataset in self.datasets:
            hidden_network = self.propagation_factory.egonet.get_hidden_network(dataset)
            degree = pd.Series(hidden_network.degree(), index=hidden_network.vs['author_id'])
            metadata = self._get_metadata(dataset)
            # merge degree with metadata
            degree = degree.to_frame('degree').join(metadata.set_index('author_id'))
            degrees[dataset] = degree

        # merge dfs with the dataset as a additional column
        degrees = pd.concat(degrees, names=['dataset', 'author_id'])
        degrees = degrees.dropna(subset='is_usual_suspect').sort_values(ascending=False, by='degree').reset_index()
        return degrees

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
        legitimacies = {}
        reputations = {}
        statuses = {}
        logger.info('Getting legitimacy, status and reputation')
        for dataset in self.datasets:
            legitimacy = self.propagation_factory.network_metrics.get_legitimacy(dataset)
            reputation = self.propagation_factory.network_metrics.get_reputation(dataset)
            status = self.propagation_factory.network_metrics.get_status(dataset)
            legitimacies[dataset] = legitimacy
            reputations[dataset] = reputation
            statuses[dataset] = status

        # merge dfs with the dataset as a additional column
        legitimacies = pd.concat(legitimacies, names=['dataset', 'author_id'])
        reputations = pd.concat(reputations, names=['dataset', 'author_id'])
        statuses = pd.concat(statuses, names=['dataset', 'author_id'])

        logger.info('Getting degrees')
        # get degrees and metadata
        degrees = self._get_degrees()
        legitimacy_figures_data = defaultdict(list)
        # group by user type and plot the egonet for each user
        for user_type, group in degrees.groupby('user_type'):
            count = 0
            reputation_figures = []
            status_figures = []
            for i, row in group.iterrows():
                if count == self.top_n:
                    break
                try:
                    reputation = reputations.loc[(row['dataset'], row['author_id'])]
                    status = statuses.loc[(row['dataset'], row['author_id'])]
                    fig = self._plot_reputation(reputation)
                    reputation_figures.append(fig)
                    fig = self._plot_status(status)
                    status_figures.append(fig)
                    legitimacy = legitimacies[row['dataset'], row['author_id']]
                    legitimacy_figures_data[user_type].append(legitimacy)

                    count += 1
                    logger.info(f'Plotted legitimacy, reputation and status for '
                                f'user {row["author_id"]} from dataset {row["dataset"]} count {count}')
                except Exception as e:
                    logger.warning(
                        f'Failed to plot legitimacy, reputation and status for user {row["author_id"]} from dataset {row["dataset"]} due to {e}')

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

    def plot_cascades(self):
        logger.info('Plotting cascades')
        conversations = self._get_conversations()
        cascades = defaultdict(list)
        for user_type, group in conversations.groupby('user_type'):
            count = 0
            for i, row in group.iterrows():
                if count == self.top_n:
                    break
                try:
                    cascade = self.propagation_factory.plot_propagation_tree(row['dataset'], row['conversation_id'])
                    cascades[user_type].append(cascade)
                    count += 1
                    logger.info(f'Plotted cascade for conversation {row["conversation_id"]} from dataset {row["dataset"]} count {count}')
                except Exception as e:
                    logger.warning(
                        f'Failed to plot cascade for conversation {row["conversation_id"]} from dataset {row["dataset"]}')

        for user_type, group in cascades.items():
            for i, cascade in enumerate(group):
                cascade.write_image(f'{self.output_dir}/{user_type}_cascade_{i}.png')

    def generate_nodes_and_edges_table(self):
        logger.info('Generating nodes and edges table')
        num_nodes_edges = {}
        for dataset in self.datasets:
            hidden_network = self.propagation_factory.egonet.get_hidden_network(dataset)
            num_nodes = hidden_network.vcount()
            num_edges = hidden_network.ecount()
            num_nodes_edges[dataset] = {'num_nodes': num_nodes, 'num_edges': num_edges,
                                        'closeness': np.mean(hidden_network.closeness())}

        num_nodes_edges = pd.DataFrame(num_nodes_edges).T
        num_nodes_edges.to_csv(self.output_dir / 'nodes_edges.csv')

    def generate_performance_table(self):
        logger.info('Generating performance table')
        results = {}
        for dataset in self.datasets:
            logger.info(f'Testing dataset {dataset}')
            results[dataset] = self._test_dataset(dataset)

        results = pd.DataFrame(results).T
        results.to_csv(self.output_dir / 'model_performance.csv')

    def _test_dataset(self, dataset):
        dataset_generator = PropagationDatasetGenerator(dataset, host=self.host, port=self.port)
        features = dataset_generator.generate_propagation_dataset()
        y = features['propagated']
        X = features.drop(columns=['propagated'])

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
        # self.plot_propagation_tree()
        # self.plot_egonet_and_backbone()
        # self.plot_legitimacy_status_and_reputation()
        # self.plot_cascades()
        # self.generate_nodes_and_edges_table()
        self.generate_performance_table()
        logger.info('Results processed')


if __name__ == '__main__':
    output_dir = Path('../results/final')
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ['MENA_Agressions', 'MENA_Ajudes', 'Openarms']
    results = Results(datasets=datasets, host='mongodb://srvinv02.esade.es', port=27017,
                      output_dir=output_dir, top_n=3)
    results.process()
