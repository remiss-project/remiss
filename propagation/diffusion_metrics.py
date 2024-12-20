import datetime
import logging

import igraph as ig
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from matplotlib import pyplot as plt
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from pymongoarrow.schema import Schema
from tqdm import tqdm

from propagation.base import BasePropagationMetrics

patch_all()

logger = logging.getLogger(__name__)


class DiffusionMetrics(BasePropagationMetrics):

    def __init__(self, egonet, host='localhost', port=27017, reference_types=('retweeted', 'quoted', 'replied_to'),
                 n_jobs=-2):
        super().__init__(host, port, reference_types)
        self.n_jobs = n_jobs
        self.egonet = egonet

    def load_cascade_data(self, dataset, cascade_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('diffusion_metrics')
        cascade_data = collection.find_one({'cascade_id': cascade_id})
        client.close()
        if not cascade_data:
            raise RuntimeError(f'Diffusion metrics for conversation {cascade_id} not found')
        return cascade_data

    def get_diffusion_metrics(self, dataset, tweet_id):
        cascade_data = self.load_cascade_data(dataset, tweet_id)
        propagation_tree = self._get_propagation_tree_from_cascade_data(cascade_data)
        size_over_time = self._get_size_over_time_from_cascade_data(cascade_data)
        depth_over_time = self._get_depth_over_time_from_cascade_data(cascade_data)
        max_breadth_over_time = self._get_max_breadth_over_time_from_cascade_data(cascade_data)
        structural_virality_over_time = self._get_structural_virality_over_time_from_cascade_data(cascade_data)
        return propagation_tree, size_over_time, depth_over_time, max_breadth_over_time, structural_virality_over_time

    def _get_propagation_tree_from_cascade_data(self, cascade_data):
        graph = ig.Graph(directed=True)
        graph.add_vertices(len(cascade_data['vs_attributes']['author_id']))
        for attribute, values in cascade_data['vs_attributes'].items():
            graph.vs[attribute] = values
        graph.add_edges(cascade_data['edges'])
        return graph

    def _get_size_over_time_from_cascade_data(self, cascade_data):
        size_over_time = pd.Series(cascade_data['size_over_time'], name='Size')
        size_over_time.index = pd.to_datetime(size_over_time.index)
        return size_over_time

    def _get_depth_over_time_from_cascade_data(self, cascade_data):
        depth_over_time = pd.Series(cascade_data['depth_over_time'], name='Depth')
        depth_over_time.index = pd.to_datetime(depth_over_time.index)
        return depth_over_time

    def _get_max_breadth_over_time_from_cascade_data(self, cascade_data):
        max_breadth_over_time = pd.Series(cascade_data['max_breadth_over_time'], name='Max Breadth')
        max_breadth_over_time.index = pd.to_datetime(max_breadth_over_time.index)
        return max_breadth_over_time

    def _get_structural_virality_over_time_from_cascade_data(self, cascade_data):
        structural_virality_over_time = pd.Series(cascade_data['structural_virality_over_time'],
                                                  name='Structural Virality')
        structural_virality_over_time.index = pd.to_datetime(structural_virality_over_time.index)
        return structural_virality_over_time

    def get_propagation_tree(self, dataset, tweet_id):
        cascade_data = self.load_cascade_data(dataset, tweet_id)
        return self._get_propagation_tree_from_cascade_data(cascade_data)

    def get_size_over_time(self, dataset, tweet_id):
        cascade_data = self.load_cascade_data(dataset, tweet_id)
        return self._get_size_over_time_from_cascade_data(cascade_data)

    def get_depth_over_time(self, dataset, tweet_id):
        cascade_data = self.load_cascade_data(dataset, tweet_id)
        return self._get_depth_over_time_from_cascade_data(cascade_data)

    def get_max_breadth_over_time(self, dataset, tweet_id):
        cascade_data = self.load_cascade_data(dataset, tweet_id)
        return self._get_max_breadth_over_time_from_cascade_data(cascade_data)

    def get_structural_virality_over_time(self, dataset, tweet_id):
        cascade_data = self.load_cascade_data(dataset, tweet_id)
        return self._get_structural_virality_over_time_from_cascade_data(cascade_data)

    def compute_propagation_tree(self, dataset, tweet_id):
        hidden_network = self.egonet.get_hidden_network(dataset)
        references = self.get_references(dataset, tweet_id)
        if references.empty:
            vertices = self._get_vertex_from_tweet(dataset, tweet_id)
        else:
            vertices = self._get_vertices_from_edges(references, tweet_id)

        hidden_network = hidden_network.subgraph(hidden_network.vs.select(author_id_in=list(vertices['author_id'])))

        propagation_tree = ig.Graph(directed=True)

        # Populate graph vertices
        for _, tweet in vertices.reset_index().iterrows():
            propagation_tree.add_vertex(name=None, **tweet)

        original_tweet_index = propagation_tree.vs.find(tweet_id=tweet_id).index

        # Populate graph edges
        for vertex in propagation_tree.vs:
            vertex_type = vertex['type']
            if vertex_type == 'replied_to':
                # No need to patch, we can use references directly
                source = references[references['target'] == vertex['tweet_id']]
                if not source.empty:
                    for _, source in source.iterrows():
                        source_vertex = propagation_tree.vs.find(tweet_id=source['source'])
                        propagation_tree.add_edge(source_vertex.index, vertex.index)
            elif vertex_type in {'retweeted', 'quoted'}:
                # Use the hidden network to patch
                neighbor_hidden_indexes = hidden_network.neighbors(
                    hidden_network.vs.find(author_id=vertex['author_id']), mode='in')
                neighbor_usernames = [hidden_network.vs[neighbor]['author_id'] for neighbor in neighbor_hidden_indexes]
                # Get all vertices that are in the hidden network and are neighbors of the current vertex
                neighbors = vertices[vertices['author_id'].isin(neighbor_usernames)]
                if not neighbors.empty:
                    # get the tweet closest in the past to the current tweet
                    neighbors = neighbors[neighbors['created_at'] < vertex['created_at']]
                    if not neighbors.empty:
                        source = neighbors.loc[neighbors['created_at'].idxmax()]
                        source_index = propagation_tree.vs.find(tweet_id=source.name).index
                        propagation_tree.add_edge(source_index, vertex.index)

        # Link the original tweet to each graph connected component
        for component in propagation_tree.connected_components(mode='weak'):
            # If the original tweet is not in the component, add it liked to the earliest tweet in the component
            if original_tweet_index not in component:
                # get node with no incoming edges
                root = [v for v in component if propagation_tree.degree(v, mode='in') == 0]
                if root:
                    propagation_tree.add_edge(original_tweet_index, root[0])
                    if len(root) > 1:
                        logger.warning(f'More than one root in component {component}')

        return propagation_tree

    def _plot_graph_igraph(self, graph):
        try:
            color_type = {'original': 'blue',
                          'retweeted': 'green',
                          'quoted': 'orange',
                          'replied_to': 'red',
                          'other': 'yellow'}
            color = [color_type[vertex['type']] for vertex in graph.vs]
        except (ValueError, KeyError):
            color = 'red'
        fig, ax = plt.subplots(figsize=(20, 20))
        layout = graph.layout('fr')
        index = list(range(graph.vcount()))
        ig.plot(graph, layout=layout, target=ax, node_size=3, vertex_label=index, arrow_size=10,
                vertex_color=color)
        plt.show()

    def get_references(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        raw = client.get_database(dataset).get_collection('raw')

        references = []
        found = False
        sources = [tweet_id]

        while not found:
            targets_pipeline = [
                {'$match': {'referenced_tweets.id': {'$in': sources}}},
                {'$unwind': '$referenced_tweets'},
                {'$match': {'$expr': {'$ne': ['$referenced_tweets.author.id', '$author.id']}}},
                {'$project': {'_id': 0, 'source': '$referenced_tweets.id', 'target': '$id',
                              'source_author_id': '$referenced_tweets.author.id',
                              'source_username': '$referenced_tweets.author.username',
                              'target_author_id': '$author.id',
                              'target_username': '$author.username',
                              'source_text': '$referenced_tweets.text',
                              'target_text': '$text',
                              'type': '$referenced_tweets.type',
                              'source_created_at': '$referenced_tweets.created_at',
                              'target_created_at': '$created_at'
                              }}
            ]
            targets = raw.aggregate_pandas_all(targets_pipeline)
            if not targets.empty:
                references.append(targets)
                sources = targets['target'].tolist()
            else:
                found = True
        if len(references) > 0:
            references = pd.concat(references, ignore_index=True)
        else:
            references = pd.DataFrame(columns=['source', 'target', 'source_author_id', 'source_username',
                                               'target_author_id', 'target_username', 'source_text', 'target_text',
                                               'type',
                                               'source_created_at', 'target_created_at'])

        client.close()
        references = references.dropna(subset=['source_created_at'])

        # Make sure created_ats are datetime objects with no timezone
        references['source_created_at'] = pd.to_datetime(references['source_created_at'], utc=True)
        references['target_created_at'] = pd.to_datetime(references['target_created_at'], utc=True)

        return references

    def _get_vertices_from_edges(self, edges, tweet_id):
        sources = edges[['source', 'source_author_id', 'source_username', 'source_text', 'source_created_at']]
        sources = sources.rename(columns={'source': 'tweet_id',
                                          'source_author_id': 'author_id',
                                          'source_username': 'username',
                                          'source_text': 'text',
                                          'source_created_at': 'created_at'})
        targets = edges[['target', 'target_author_id', 'target_username', 'target_text', 'type', 'target_created_at']]
        targets = targets.rename(columns={'target': 'tweet_id',
                                          'target_author_id': 'author_id',
                                          'target_username': 'username',
                                          'target_text': 'text',
                                          'target_created_at': 'created_at'})
        vertices = pd.concat([sources, targets], ignore_index=True)
        vertices = vertices.sort_values('type', ascending=False)

        vertices = vertices.drop_duplicates(subset='tweet_id').sort_values('created_at').set_index('tweet_id')
        vertices.loc[tweet_id, 'type'] = 'original'
        vertices['type'] = vertices['type'].fillna('other')

        return vertices

    def _get_vertex_from_tweet(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        raw = client.get_database(dataset).get_collection('raw')
        tweet = raw.find_one({'id': tweet_id})
        client.close()
        if not tweet:
            raise RuntimeError(f'Tweet {tweet_id} not found in dataset {dataset}')
        # ['tweet_id', 'author_id', 'username', 'text', 'created_at', 'type']
        vertex = pd.DataFrame([{'tweet_id': tweet['id'],
                                'author_id': tweet['author']['id'],
                                'username': tweet['author']['username'],
                                'text': tweet['text'],
                                'created_at': tweet['created_at'],
                                'type': 'original'}])
        return vertex

    def compute_diffusion_metrics(self, dataset, tweet_id):
        graph = self.compute_propagation_tree(dataset, tweet_id)
        if graph.vcount() > 1:
            shortest_paths = self.get_shortest_paths_to_original_tweet_over_time(graph)

            size_over_time = self.compute_size_over_time(graph)
            depth_over_time = self.compute_depth_over_time(shortest_paths)
            max_breadth_over_time = self.compute_max_breadth_over_time(shortest_paths)
            structural_virality_over_time = self.compute_structural_virality_over_time(graph)

        else:
            size_over_time = pd.Series([1], index=[graph.vs['created_at'][0]])
            depth_over_time = pd.Series([0], index=[graph.vs['created_at'][0]])
            max_breadth_over_time = pd.Series([1], index=[graph.vs['created_at'][0]])
            structural_virality_over_time = pd.Series([0], index=[graph.vs['created_at'][0]])

        return graph, size_over_time, depth_over_time, max_breadth_over_time, structural_virality_over_time

    def compute_size_over_time(self, graph):
        # get the difference between the first tweet and the rest in minutes
        size = pd.Series(np.ones(graph.vcount(), dtype=int), index=graph.vs['created_at'], name='Size').sort_index()
        size = size.cumsum()
        # Remove duplicated timestamps
        size = size.groupby(size.index).max()

        return size

    def compute_depth_over_time(self, shortest_paths):
        depths = shortest_paths.cummax()
        depths = depths.rename('Depth')
        depths = depths[~depths.index.duplicated(keep='first')]
        return depths

    def compute_max_breadth_over_time(self, shortest_paths):
        max_breadth = {}
        for i, time in enumerate(shortest_paths.index):
            max_breadth[time] = shortest_paths.iloc[:i + 1].value_counts().max()

        max_breadth = pd.Series(max_breadth, name='Max Breadth')
        return max_breadth

    def compute_structural_virality_over_time(self, graph):
        # Specifically, we define structural
        # virality as the average distance between all pairs
        # of nodes in a diffusion tree
        shortests_paths = pd.DataFrame(graph.as_undirected().shortest_paths_dijkstra())
        shortests_paths = shortests_paths.replace(float('inf'), pd.NA)
        created_at = pd.Series(graph.vs['created_at'])
        created_at = created_at.dropna()
        order = created_at.argsort()
        shortests_paths = shortests_paths.iloc[order, order]
        created_at = created_at.iloc[order]
        structured_virality = {}
        for i, time in enumerate(created_at):
            current_shortests_paths = shortests_paths.iloc[:i + 1, :i + 1]
            structured_virality[time] = current_shortests_paths.mean().mean()

        structured_virality = pd.Series(structured_virality, name='Structural Virality')
        return structured_virality

    def get_size_cascade_ccdf(self, dataset):
        return self._load_plot_size_cascade_ccdf_from_mongodb(dataset)

    def get_cascade_count_over_time(self, dataset):
        return self._load_plot_cascade_count_over_time_from_mongodb(dataset)

    def compute_size_cascade_ccdf(self, dataset):
        conversation_sizes = self.get_cascade_sizes(dataset)
        conversation_sizes['user_type'] = conversation_sizes.apply(transform_user_type, axis=1)
        conversation_sizes = conversation_sizes.drop(columns=['is_usual_suspect', 'party'])
        ccdf = {}
        for user_type, df in conversation_sizes.groupby('user_type'):
            ccdf[user_type] = df['size'].value_counts(normalize=True).sort_index(ascending=False).cumsum()

        ccdf = pd.DataFrame(ccdf)
        ccdf = ccdf * 100
        return ccdf


    def get_cascade_sizes(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # Compute the amount of times that a tweet that has no referenced_tweets is referenced
        pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$project': {'_id': 0, 'original_id': '$id',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          }}
        ]
        schema = Schema({'original_id': str, 'is_usual_suspect': bool, 'party': str})
        cascades = collection.aggregate_pandas_all(pipeline, schema=schema)

        original_ids = cascades['original_id'].tolist()
        cascades = cascades.set_index('original_id')
        cascade_size_pipeline = [
            {'$match': {'referenced_tweets.id': {'$in': original_ids}}},
            {'$unwind': '$referenced_tweets'},
            {'$group': {'_id': '$referenced_tweets.id', 'size': {'$count': {}}}},
            {'$project': {'_id': 0, 'cascade_id': '$_id', 'size': 1}}
        ]
        schema = Schema({'cascade_id': str, 'size': int})
        cascade_sizes = collection.aggregate_pandas_all(cascade_size_pipeline, schema=schema)
        client.close()

        cascades = cascades.join(cascade_sizes.set_index('cascade_id'))
        cascades = cascades.dropna(subset=['size'])

        return cascades

    def compute_cascade_count_over_time(self, dataset):
        logger.info(f'Computing cascade count over time for dataset {dataset}')
        cascade_ids = self.get_cascade_ids(dataset)
        logger.info(f'Found {len(cascade_ids)} cascades in dataset {dataset}. Sampling')
        cascade_ids = cascade_ids.fillna(1)
        cascade_ids = cascade_ids.set_index('created_at')
        cascade_ids_week = cascade_ids.resample('W').count()
        if cascade_ids_week.shape[0] <= 20:
            cascade_ids = cascade_ids.resample('D').count()
        else:
            cascade_ids = cascade_ids_week
        cascade_ids = cascade_ids.rename(columns={'tweet_id': 'Cascade Count'})
        cascade_ids = cascade_ids['Cascade Count']
        return cascade_ids

    def get_tweet(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        tweet = collection.find_one({'id': tweet_id})
        client.close()
        if tweet:
            return tweet
        else:
            raise RuntimeError(f'Tweet {tweet_id} not found in dataset {dataset}')

    @staticmethod
    def get_shortest_paths_to_original_tweet_over_time(graph):
        # graph = graph.as_undirected()
        original_tweet_id = graph.vs.find(type='original')['tweet_id']
        original_tweet_index = graph.vs.find(tweet_id=original_tweet_id).index
        # shortest_paths = pd.Series(graph.shortest_paths_dijkstra(source=original_tweet_index)[0],
        #                            index=graph.vs['created_at'])

        shortest_paths = pd.Series(graph.get_shortest_paths(0), index=graph.vs['created_at']).apply(
            lambda x: len(x) - 1)
        # Drop infinity
        shortest_paths = shortest_paths.replace(float('inf'), pd.NA)
        shortest_paths = shortest_paths.dropna()
        shortest_paths = shortest_paths.sort_index()

        return shortest_paths

    def get_cascade_ids(self, dataset):
        logger.info(f'Getting cascades in dataset {dataset}')
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # get tweets that are the start of the cascade, so they are not referenced by any other tweet
        original_tweets_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$group': {'_id': '$id',
                        'created_at': {'$first': '$created_at'},
                        'count': {'$count': {}}}},
            {'$project': {'_id': 0, 'tweet_id': '$_id', 'created_at': 1, 'count': 1}}
        ]
        schema = Schema({'tweet_id': str, 'created_at': datetime.datetime})

        original_tweets = collection.aggregate_pandas_all(original_tweets_pipeline, schema=schema)

        # get tweets that are referenced by other tweets
        referenced_tweets_pipeline = [
            {'$match': {'referenced_tweets': {'$exists': True}}},
            {'$unwind': '$referenced_tweets'},
            {'$group': {'_id': '$referenced_tweets.id'}},
            {'$project': {'_id': 0, 'tweet_id': '$_id'}}
        ]
        schema = Schema({'tweet_id': str})
        referenced_tweets = collection.aggregate_pandas_all(referenced_tweets_pipeline, schema=schema)
        client.close()

        # We only want the original tweets that are referenced by other tweets
        df = original_tweets[original_tweets['tweet_id'].isin(referenced_tweets['tweet_id'])]
        return df

    def get_cascade_size(self, dataset, tweet_id):
        try:
            graph = self.get_propagation_tree(dataset, tweet_id)
        except RuntimeError as e:
            if 'not found in dataset' in str(e):
                # The tweet is not in the dataset, so we can't compute the cascade size
                raise e
            logger.error(f'Error getting cascade size for {tweet_id}: {e}. Recomputing.')
            graph = self.compute_propagation_tree(dataset, tweet_id)

        return graph.vcount()

    def persist(self, datasets):
        self.persist_diffusion_metrics(datasets)
        self.persist_diffusion_static_plots(datasets)

    def persist_diffusion_metrics(self, datasets, max_cascades=None, erase_existing=False):
        n_jobs = self.n_jobs
        self.n_jobs = 1

        for dataset in datasets:
            logger.info(f'Persisting diffusion metrics for {dataset}')
            start_time = datetime.datetime.now()
            client = MongoClient(self.host, self.port)
            database = client.get_database(dataset)
            collection = database.get_collection('diffusion_metrics')
            if erase_existing:
                collection.drop()

            cascade_ids = self.get_cascade_by_retweet_count(dataset, max_cascades)
            if cascade_ids.empty:
                logger.warning(f'No cascades found for dataset {dataset}')
            else:
                jobs = []
                for cascade_id in tqdm(cascade_ids['tweet_id']):
                    # if not self.has_diffusion_metrics(dataset, cascade_id):
                    jobs.append(delayed(self._compute_cascade_metrics_for_persistence)(dataset, cascade_id))

                cascade_data = Parallel(n_jobs=n_jobs, backend='threading', verbose=10)(jobs)

                collection.insert_many(cascade_data)
                client.close()
                logger.info(f'Finished persisting diffusion metrics for {dataset}. '
                            f'Time elapsed: {datetime.datetime.now() - start_time}')

        self.n_jobs = n_jobs

    def has_diffusion_metrics(self, dataset, cascade_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('diffusion_metrics')
        cascade_data = collection.find_one({'cascade_id': cascade_id})
        client.close()
        return cascade_data is not None

    def get_cascade_by_retweet_count(self, dataset, max_cascades=None):
        pipeline_initial = [
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$sort': {'public_metrics.retweet_count': -1}},
            {'$project': {
                '_id': 0,
                'tweet_id': '$id',
                'retweet_count': '$public_metrics.retweet_count'}},
        ]
        if max_cascades is not None:
            logger.info(f'Limiting cascades to {max_cascades} for dataset {dataset}')
            pipeline_initial.append({'$limit': max_cascades})
        schema = Schema({'tweet_id': str, 'retweet_count': int})
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        cascade_ids = collection.aggregate_pandas_all(pipeline_initial, schema=schema)
        client.close()
        return cascade_ids

    def persist_diffusion_static_plots(self, datasets):
        for dataset in datasets:
            start_time = datetime.datetime.now()
            logger.info(f'Persisting diffusion static plots for {dataset}')
            self._persist_plot_size_cascade_ccdf_to_mongodb(dataset)
            self._persist_plot_cascade_count_over_time_to_mongodb(dataset)
            logger.info(f'Finished persisting diffusion static plots for {dataset}. '
                        f'Time elapsed: {datetime.datetime.now() - start_time}')

    def _compute_cascade_metrics_for_persistence(self, dataset, cascade_id):
        try:
            graph, size_over_time, depth_over_time, max_breadth_over_time, structural_virality_over_time = \
                self.compute_diffusion_metrics(dataset, cascade_id)
            try:
                size_over_time = size_over_time.to_dict()
                size_over_time = {str(key): value for key, value in size_over_time.items()}
            except Exception as e:
                logger.error(f'Error converting {cascade_id} size over time to json: {e}')
                size_over_time = None

            try:
                depth_over_time = depth_over_time.to_dict()
                depth_over_time = {str(key): value for key, value in depth_over_time.items()}
            except Exception as e:
                logger.error(f'Error converting {cascade_id} depth over time to json: {e}')
                depth_over_time = None

            try:
                max_breadth_over_time = max_breadth_over_time.to_dict()
                max_breadth_over_time = {str(key): value for key, value in max_breadth_over_time.items()}
            except Exception as e:
                logger.error(f'Error converting {cascade_id} max breadth over time to json: {e}')
                max_breadth_over_time = None

            try:
                structural_virality_over_time = structural_virality_over_time.to_dict()
                structural_virality_over_time = {str(key): value for key, value in
                                                 structural_virality_over_time.items()}
            except Exception as e:
                logger.error(f'Error converting {cascade_id} structural virality over time to json: {e}')
                structural_virality_over_time = None

            attributes = {attribute: graph.vs[attribute] for attribute in graph.vs.attributes()}

            return {'cascade_id': cascade_id,
                    'edges': graph.get_edgelist(),
                    'vs_attributes': attributes,
                    'size_over_time': size_over_time,
                    'depth_over_time': depth_over_time,
                    'max_breadth_over_time': max_breadth_over_time,
                    'structural_virality_over_time': structural_virality_over_time}

        except Exception as e:
            logger.error(f'Error processing conversation {cascade_id}: {e}')

            raise e

    def _persist_plot_size_cascade_ccdf_to_mongodb(self, dataset):
        logger.info(f'Persisting size cascade ccdf for {dataset}')
        size_cascade = self.compute_size_cascade_ccdf(dataset)
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('size_cascade_ccdf')
        collection.drop()
        logger.info(f'Inserting {len(size_cascade)} records')
        collection.insert_many(size_cascade.reset_index().to_dict(orient='records'))
        client.close()

    def _persist_plot_cascade_count_over_time_to_mongodb(self, dataset):
        logger.info(f'Persisting cascade count over time for {dataset}')
        cascade_count_over_time = self.compute_cascade_count_over_time(dataset)
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('cascade_count_over_time')
        collection.drop()
        logger.info(f'Inserting {len(cascade_count_over_time)} records')
        collection.insert_many(cascade_count_over_time.reset_index().to_dict(orient='records'))
        client.close()

    def _load_plot_size_cascade_ccdf_from_mongodb(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('size_cascade_ccdf')
        size_cascade = collection.aggregate_pandas_all([{'$project': {'_id': 0}}])
        client.close()
        return size_cascade.set_index('size')

    def _load_plot_cascade_count_over_time_from_mongodb(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('cascade_count_over_time')
        cascade_count_over_time = collection.aggregate_pandas_all([{'$project': {'_id': 0}}])
        client.close()
        return cascade_count_over_time.set_index('created_at')['Cascade Count']


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
