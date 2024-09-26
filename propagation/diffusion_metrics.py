import datetime
import logging

import igraph as ig
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from pymongoarrow.schema import Schema
from tqdm import tqdm

from propagation.base import BasePropagationMetrics

patch_all()

logger = logging.getLogger(__name__)


class DiffusionMetrics(BasePropagationMetrics):

    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted', 'replied_to'), n_jobs=-2):
        super().__init__(host, port, reference_types)
        self.n_jobs = n_jobs

    def load_conversation_data(self, dataset, cascade_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('diffusion_metrics')
        cascade_data = collection.find_one({'cascade_id': cascade_id})
        client.close()
        if not cascade_data:
            raise RuntimeError(f'Diffusion metrics for conversation {cascade_id} not found')
        return cascade_data

    def get_propagation_tree(self, dataset, tweet_id):
        conversation_data = self.load_conversation_data(dataset, tweet_id)
        graph = ig.Graph(directed=True)
        graph.add_vertices(len(conversation_data['vs_attributes']['author_id']))
        for attribute, values in conversation_data['vs_attributes'].items():
            graph.vs[attribute] = values
        graph.add_edges(conversation_data['edges'])
        return graph

    def get_size_over_time(self, dataset, tweet_id):
        conversation_data = self.load_conversation_data(dataset, tweet_id)
        size_over_time = pd.Series(conversation_data['size_over_time'], name='Size')
        size_over_time.index = pd.to_datetime(size_over_time.index)
        return size_over_time

    def get_depth_over_time(self, dataset, tweet_id):
        conversation_data = self.load_conversation_data(dataset, tweet_id)
        depth_over_time = pd.Series(conversation_data['depth_over_time'], name='Depth')
        depth_over_time.index = pd.to_datetime(depth_over_time.index)
        return depth_over_time

    def get_max_breadth_over_time(self, dataset, tweet_id):
        conversation_data = self.load_conversation_data(dataset, tweet_id)
        max_breadth_over_time = pd.Series(conversation_data['max_breadth_over_time'], name='Max Breadth')
        max_breadth_over_time.index = pd.to_datetime(max_breadth_over_time.index)
        return max_breadth_over_time

    def get_structural_virality_over_time(self, dataset, tweet_id):
        conversation_data = self.load_conversation_data(dataset, tweet_id)
        structural_virality_over_time = pd.Series(conversation_data['structural_virality_over_time'],
                                                  name='Structural Virality')
        structural_virality_over_time.index = pd.to_datetime(structural_virality_over_time.index)
        return structural_virality_over_time

    def compute_propagation_tree(self, dataset, tweet_id):
        references = self.get_references(dataset, tweet_id)
        if references.empty:
            vertices = self._get_vertex_from_tweet(dataset, tweet_id)
        else:
            vertices = self._get_vertices_from_edges(references)
        graph = self._get_graph(vertices, references)

        return graph

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

        return references

    def _get_graph(self, vertices, edges):
        graph = ig.Graph(directed=True)

        graph.add_vertices(len(vertices))

        for column in vertices.columns:
            graph.vs[column] = vertices[column].tolist()

        if len(edges) > 0:
            # switch id by position (which will be the node id in the graph) and set it as index
            tweet_to_id = vertices['tweet_id'].reset_index().set_index('tweet_id')
            # convert references which are author id based to graph id based
            edges['source'] = tweet_to_id.loc[edges['source']].reset_index(drop=True)
            edges['target'] = tweet_to_id.loc[edges['target']].reset_index(drop=True)

            graph.add_edges(edges[['source', 'target']].to_records(index=False).tolist())

        return graph

    def _get_vertices_from_edges(self, edges):
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

        vertices = vertices.drop_duplicates(subset='tweet_id').sort_values('created_at').reset_index(drop=True)

        vertices['type'] = vertices['type'].fillna('original')
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

    def compute_size_over_time(self, graph):
        # get the difference between the first tweet and the rest in minutes
        size = pd.Series(np.ones(graph.vcount(), dtype=int), index=graph.vs['created_at'], name='Size').sort_index()
        size = size.cumsum()
        # Remove duplicated timestamps
        size = size.groupby(size.index).max()

        return size

    def compute_depth_over_time(self, graph):
        shortest_paths = self.get_shortest_paths_to_original_tweet(graph)
        created_at = pd.Series(graph.vs['created_at'], name='Depth')
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        depths = {}
        for i, time in enumerate(created_at):
            depths[time] = shortest_paths.iloc[:i + 1].max()

        depths = pd.Series(depths, name='Depth')
        return depths

    def compute_max_breadth_over_time(self, graph):
        shortest_paths = self.get_shortest_paths_to_original_tweet(graph)
        created_at = pd.Series(graph.vs['created_at'], name='Max Breadth')
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        max_breadth = {}
        for i, time in enumerate(created_at):
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
        client.close()

        original_ids = cascades['original_id'].tolist()
        cascades = cascades.set_index('original_id')
        jobs = [delayed(self.get_references)(dataset, original_id) for original_id in original_ids]
        references = Parallel(n_jobs=self.n_jobs, backend='threading', verbose=0)(jobs)
        cascades['size'] = [len(cascade) for cascade in references]

        return cascades

    def get_cascade_count_over_time(self, dataset):
        conversation_ids = self.get_cascade_ids(dataset)
        conversation_ids = conversation_ids.set_index('created_at')
        conversation_ids = conversation_ids.resample('ME').count()
        conversation_ids = conversation_ids.fillna(0)
        conversation_ids = conversation_ids.rename(columns={'tweet_id': 'Cascade Count'})
        conversation_ids = conversation_ids['Cascade Count']
        return conversation_ids

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
    def get_shortest_paths_to_original_tweet(graph):
        original_tweet_id = graph.vs.find(type='original')['tweet_id']
        original_tweet_index = graph.vs.find(tweet_id=original_tweet_id).index
        shortest_paths = pd.Series(graph.shortest_paths_dijkstra(source=original_tweet_index)[0])
        shortest_paths = shortest_paths.replace(float('inf'), pd.NA)
        return shortest_paths

    def get_cascade_ids(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # get tweets that are the start of the cascade, so they are not referenced by any other tweet
        pipeline = [
            {'$match': {'referenced_tweets': {'$exists': False}}},
            {'$group': {'_id': '$id', 'created_at': {'$first': '$created_at'}}},
            {'$project': {'_id': 0, 'tweet_id': '$_id', 'created_at': 1}}
        ]
        schema = Schema({'tweet_id': str, 'created_at': datetime.datetime})
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        client.close()
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
        jobs = []
        n_jobs = self.n_jobs
        self.n_jobs = 1
        for dataset in datasets:
            cascade_ids = self.get_cascade_ids(dataset)
            for cascade_id in tqdm(cascade_ids['tweet_id']):
                jobs.append(delayed(self._persist_conversation_metrics)(dataset, cascade_id))

            cascade_data = Parallel(n_jobs=n_jobs, backend='threading', verbose=10)(jobs)
            client = MongoClient(self.host, self.port)
            database = client.get_database(dataset)
            collection = database.get_collection('diffusion_metrics')
            collection.insert_many(cascade_data)

        self.n_jobs = n_jobs

    def _persist_conversation_metrics(self, dataset, cascade_id):
        try:
            graph = self.compute_propagation_tree(dataset, cascade_id)
            size_over_time = self.compute_size_over_time(graph)
            depth_over_time = self.compute_depth_over_time(graph)
            max_breadth_over_time = self.compute_max_breadth_over_time(graph)
            structural_virality_over_time = self.compute_structural_virality_over_time(graph)

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


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
