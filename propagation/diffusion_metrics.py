import datetime
import logging

import igraph as ig
import numpy as np
import pandas as pd
import pymongoarrow
from joblib import delayed, Parallel
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from tqdm import tqdm

from propagation.base import BasePropagationMetrics

pymongoarrow.monkey.patch_all()

logger = logging.getLogger(__name__)


class DiffusionMetrics(BasePropagationMetrics):

    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted')):
        super().__init__(host, port, reference_types)
        self._propagation_tree = {}
        self._size_over_time = {}
        self._depth_over_time = {}
        self._max_breadth_over_time = {}
        self._structural_virality_over_time = {}

    def get_propagation_tree(self, dataset, tweet_id):
        conversation_id = self.get_conversation_id(dataset, tweet_id)
        if (dataset, conversation_id) not in self._propagation_tree:
            # conversation_id tweet might not be present in the dataset, so we will use the tweet_id
            # TODO think of a more clever way to do this
            self._propagation_tree[(dataset, conversation_id)] = self.compute_propagation_tree(dataset, tweet_id)
        return self._propagation_tree[(dataset, conversation_id)]

    def get_size_over_time(self, dataset, tweet_id):
        conversation_id = self.get_conversation_id(dataset, tweet_id)
        if (dataset, conversation_id) not in self._size_over_time:
            self._size_over_time[(dataset, conversation_id)] = self.compute_size_over_time(dataset, tweet_id)
        return self._size_over_time[(dataset, conversation_id)]

    def get_depth_over_time(self, dataset, tweet_id):
        conversation_id = self.get_conversation_id(dataset, tweet_id)
        if (dataset, conversation_id) not in self._depth_over_time:
            self._depth_over_time[(dataset, conversation_id)] = self.compute_depth_over_time(dataset, tweet_id)
        return self._depth_over_time[(dataset, conversation_id)]

    def get_max_breadth_over_time(self, dataset, tweet_id):
        conversation_id = self.get_conversation_id(dataset, tweet_id)
        if (dataset, conversation_id) not in self._max_breadth_over_time:
            self._max_breadth_over_time[(dataset, conversation_id)] = self.compute_max_breadth_over_time(dataset,
                                                                                                         tweet_id)
        return self._max_breadth_over_time[(dataset, conversation_id)]

    def get_structural_virality_over_time(self, dataset, tweet_id):
        conversation_id = self.get_conversation_id(dataset, tweet_id)
        if (dataset, conversation_id) not in self._structural_virality_over_time:
            self._structural_virality_over_time[
                (dataset, conversation_id)] = self.compute_structural_virality_over_time(dataset, conversation_id)
        return self._structural_virality_over_time[(dataset, conversation_id)]

    def compute_propagation_tree(self, dataset, tweet_id):
        conversation_id, conversation_tweets, references = self.get_conversation(dataset, tweet_id)
        graph = self._get_graph(conversation_tweets, references)
        self.ensure_conversation_id(conversation_id, graph)

        return graph

    def get_conversation(self, dataset, tweet_id):
        tweet = self.get_tweet(dataset, tweet_id)
        conversation_id = tweet['conversation_id']

        # get all tweets belonging to the conversation
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        references_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'source': '$id', 'target': '$referenced_tweets.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target'}},

        ]

        references = collection.aggregate_pandas_all(references_pipeline)

        tweet_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$project': {'tweet_id': '$id',
                          'author_id': '$author.id',
                          'created_at': 1
                          }}
        ]
        referenced_tweet_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.id': {'$exists': True},
                        'referenced_tweets.author': {'$exists': True},
                        'created_at': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'tweet_id': '$referenced_tweets.id',
                          'author_id': '$referenced_tweets.author.id',
                          'created_at': '$referenced_tweets.created_at'
                          }}
        ]
        merge_tweets_and_references_pipeline = [
            {'$group': {'_id': '$tweet_id',
                        'author_id': {'$first': '$author_id'},
                        'created_at': {'$first': '$created_at'},
                        }
             },
            {'$project': {'_id': 0,
                          'tweet_id': '$_id',
                          'author_id': 1,
                          'created_at': {'$dateFromString': {'dateString': '$created_at', 'onError': '$created_at'}}
                          }
             }
        ]
        tweets_pipeline = [
            *referenced_tweet_pipeline,
            {'$unionWith': {'coll': 'raw', 'pipeline': tweet_pipeline}},  # Fetch missing tweets
            *merge_tweets_and_references_pipeline
        ]
        schema = Schema({'tweet_id': str, 'author_id': str, 'created_at': datetime.datetime})
        tweets = collection.aggregate_pandas_all(tweets_pipeline, schema=schema)
        if conversation_id not in tweets['tweet_id'].values:
            # add dummy conversation id tweet picking data from the original tweet
            conversation_tweet = {'tweet_id': conversation_id,
                                  'author_id': '-',
                                  'created_at': tweets['created_at'].min() - pd.Timedelta(1, 's')}

            # link it to the oldest tweet in the conversation
            oldest_tweet_id = tweets['tweet_id'].iloc[tweets['created_at'].idxmin()]
            references = pd.concat([pd.DataFrame([{'source': conversation_id, 'target': oldest_tweet_id}]),
                                    references], ignore_index=True)
            tweets = pd.concat([pd.DataFrame([conversation_tweet]), tweets], ignore_index=True)
        client.close()

        return conversation_id, tweets, references

    def get_conversation_id(self, dataset, tweet_id):
        tweet = self.get_tweet(dataset, tweet_id)
        return tweet['conversation_id']

    def get_conversation_sizes(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # Compute the length of the conversation for conversation
        pipeline = [
            {'$match': {'conversation_id': {'$exists': True}}},
            {'$group': {'_id': '$conversation_id',
                        'size': {'$count': {}},
                        'is_usual_suspect': {'$addToSet': '$author.remiss_metadata.is_usual_suspect'},
                        'party': {'$addToSet': '$author.remiss_metadata.party'}
                        }},
            {'$sort': {'size': -1}},
            {'$project': {'_id': 0, 'conversation_id': '$_id', 'size': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]}
                          }}
        ]
        schema = Schema({'conversation_id': str, 'size': int, 'is_usual_suspect': bool, 'party': str})
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        return df

    def compute_size_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # get the difference between the first tweet and the rest in minutes
        size = pd.Series(np.ones(graph.vcount(), dtype=int), index=graph.vs['created_at'], name='Size').sort_index()
        size = size.cumsum()
        # Remove duplicated timestamps
        size = size.groupby(size.index).max()

        return size

    def compute_depth_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        shortest_paths = self.get_shortest_paths_to_conversation_id(graph)
        created_at = pd.Series(graph.vs['created_at'], name='Depth')
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        depths = {}
        for i, time in enumerate(created_at):
            depths[time] = shortest_paths.iloc[:i + 1].max()

        depths = pd.Series(depths, name='Depth')
        return depths

    def compute_max_breadth_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        shortest_paths = self.get_shortest_paths_to_conversation_id(graph)
        created_at = pd.Series(graph.vs['created_at'], name='Max Breadth')
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        max_breadth = {}
        for i, time in enumerate(created_at):
            max_breadth[time] = shortest_paths.iloc[:i + 1].value_counts().max()

        max_breadth = pd.Series(max_breadth, name='Max Breadth')
        return max_breadth

    def get_structural_virality(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # Specifically, we define structural
        # virality as the average distance between all pairs
        # of nodes in a diffusion tree
        shortests_paths = pd.DataFrame(graph.as_undirected().shortest_paths_dijkstra())
        shortests_paths = shortests_paths.replace(float('inf'), pd.NA)
        structured_virality = shortests_paths.mean().mean()
        return structured_virality

    def get_structural_virality_and_timespan(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # Specifically, we define structural
        # virality as the average distance between all pairs
        # of nodes in a diffusion tree
        shortests_paths = pd.DataFrame(graph.as_undirected().shortest_paths_dijkstra())
        shortests_paths = shortests_paths.replace(float('inf'), pd.NA)
        created_at = pd.Series(graph.vs['created_at'])
        structured_virality = shortests_paths.mean().mean()
        timespan = created_at.max() - created_at.min()
        return structured_virality, timespan

    def get_structural_viralities(self, dataset):
        conversation_ids = self.get_conversation_ids(dataset)
        structured_viralities = []
        timespans = []
        for conversation_id in conversation_ids['conversation_id']:
            structured_virality, timespan = self.get_structural_virality_and_timespan(dataset, conversation_id)
            structured_viralities.append(structured_virality)
            timespans.append(timespan)
        # Cast to pandas
        structured_viralities = pd.DataFrame({'conversation_id': conversation_ids['conversation_id'],
                                              'structured_virality': structured_viralities,
                                              'timespan': timespans})
        structured_viralities = structured_viralities.set_index('conversation_id')

        return structured_viralities

    def compute_structural_virality_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
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

    def get_depth_cascade_ccdf(self, dataset):
        conversation_ids = self.get_conversation_ids(dataset)
        depths = []
        for conversation_id in conversation_ids['conversation_id']:
            graph = self.get_propagation_tree(dataset, conversation_id)
            depth = self.get_shortest_paths_to_conversation_id(graph).max()
            depths.append(depth)

        conversation_ids['depth'] = depths
        conversation_ids['user_type'] = conversation_ids.apply(transform_user_type, axis=1)
        conversation_ids = conversation_ids.drop(columns=['is_usual_suspect', 'party'])
        ccdf = {}
        for user_type, df in conversation_ids.groupby('user_type'):
            ccdf[user_type] = df['depth'].value_counts(normalize=True).sort_index(ascending=False).cumsum()

        ccdf = pd.DataFrame(ccdf)
        ccdf = ccdf * 100
        return ccdf

    def get_size_cascade_ccdf(self, dataset):
        conversation_sizes = self.get_conversation_sizes(dataset)
        conversation_sizes['user_type'] = conversation_sizes.apply(transform_user_type, axis=1)
        conversation_sizes = conversation_sizes.drop(columns=['is_usual_suspect', 'party'])
        ccdf = {}
        for user_type, df in conversation_sizes.groupby('user_type'):
            ccdf[user_type] = df['size'].value_counts(normalize=True).sort_index(ascending=False).cumsum()

        ccdf = pd.DataFrame(ccdf)
        ccdf = ccdf * 100
        return ccdf

    def get_cascade_count_over_time(self, dataset):
        conversation_ids = self.get_conversation_ids(dataset)
        conversation_ids = conversation_ids.set_index('created_at')
        conversation_ids = conversation_ids.resample('ME').count()
        conversation_ids = conversation_ids.fillna(0)
        conversation_ids = conversation_ids.rename(columns={'conversation_id': 'Cascade Count'})
        conversation_ids = conversation_ids['Cascade Count']
        return conversation_ids

    def _get_graph(self, vertices, edges):
        graph = ig.Graph(directed=True)

        graph.add_vertices(len(vertices))
        graph.vs['author_id'] = vertices['author_id'].tolist()
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

    def ensure_conversation_id(self, conversation_id, graph):
        # link connected components to the conversation id vertex
        graph['conversation_id'] = conversation_id

        components = graph.connected_components(mode='weak')
        if len(components) > 1:
            conversation_id_index = graph.vs.find(tweet_id=conversation_id).index
            created_at = pd.Series(graph.vs['created_at'])
            for component in components:
                if conversation_id_index not in component:
                    # get first tweet in the component by created_at
                    first = created_at.iloc[component].idxmin()
                    # link it to the conversation id
                    graph.add_edge(conversation_id_index, first)

    def get_tweet(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        tweet = collection.find_one({'id': tweet_id})
        client.close()
        if tweet:
            return tweet
        else:
            raise RuntimeError(f'Tweet {tweet_id} not found')

    @staticmethod
    def get_shortest_paths_to_conversation_id(graph):
        conversation_id_index = graph.vs.find(tweet_id=graph['conversation_id']).index
        shortest_paths = pd.Series(graph.as_undirected().shortest_paths_dijkstra(source=conversation_id_index)[0])
        shortest_paths = shortest_paths.replace(float('inf'), pd.NA)
        return shortest_paths

    def get_conversation_ids(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # get tweets that are the start of the conversation, so its conversation_id is the tweet_id
        pipeline = [
            {'$match': {'$expr': {'$eq': ['$id', '$conversation_id']}}},
            {'$project': {'_id': 0, 'conversation_id': 1,
                          'created_at': 1
                          }}
        ]
        schema = Schema({'conversation_id': str, 'created_at': datetime.datetime})
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        client.close()
        return df

    def get_cascade_size(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        return graph.vcount()

    def persist(self, datasets):
        jobs = []
        for dataset in datasets:
            conversation_ids = self.get_conversation_ids(dataset)
            for conversation_id in tqdm(conversation_ids['conversation_id']):
                jobs.append(delayed(self._persist_conversation_metrics)(dataset, conversation_id))
        Parallel(n_jobs=-2, backend='threading')(jobs)

    def _persist_conversation_metrics(self, dataset, conversation_id):
        graph = self.get_propagation_tree(dataset, conversation_id)
        size_over_time = self.get_size_over_time(dataset, conversation_id)
        depth_over_time = self.get_depth_over_time(dataset, conversation_id)
        max_breadth_over_time = self.get_max_breadth_over_time(dataset, conversation_id)
        structural_virality_over_time = self.get_structural_virality_over_time(dataset, conversation_id)

        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('diffusion_metrics')
        try:
            size_over_time = size_over_time.to_dict()
            size_over_time = {str(key): value for key, value in size_over_time.items()}
        except Exception as e:
            logger.error(f'Error converting {conversation_id} size over time to json: {e}')
            size_over_time = None

        try:
            depth_over_time = depth_over_time.to_dict()
            depth_over_time = {str(key): value for key, value in depth_over_time.items()}
        except Exception as e:
            logger.error(f'Error converting {conversation_id} depth over time to json: {e}')
            depth_over_time = None

        try:
            max_breadth_over_time = max_breadth_over_time.to_dict()
            max_breadth_over_time = {str(key): value for key, value in max_breadth_over_time.items()}
        except Exception as e:
            logger.error(f'Error converting {conversation_id} max breadth over time to json: {e}')
            max_breadth_over_time = None

        try:
            structural_virality_over_time = structural_virality_over_time.to_dict()
            structural_virality_over_time = {str(key): value for key, value in structural_virality_over_time.items()}
        except Exception as e:
            logger.error(f'Error converting {conversation_id} structural virality over time to json: {e}')
            structural_virality_over_time = None

        attributes = {attribute: graph.vs[attribute] for attribute in graph.vs.attributes()}
        collection.insert_one({'conversation_id': conversation_id,
                               'edges': graph.get_edgelist(),
                               'vs_attributes': attributes,
                               'size_over_time': size_over_time,
                               'depth_over_time': depth_over_time,
                               'max_breadth_over_time': max_breadth_over_time,
                               'structural_virality_over_time': structural_virality_over_time})
        client.close()

    def load_from_mongodb(self, datasets):
        for dataset in datasets:
            self._load_metrics(dataset)

    def _load_metrics(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('diffusion_metrics')
        for document in collection.find():
            conversation_id = document['conversation_id']
            graph = ig.Graph(directed=True)
            graph.add_vertices(len(document['vs_attributes']['author_id']))
            for attribute, values in document['vs_attributes'].items():
                graph.vs[attribute] = values
            graph.add_edges(document['edges'])
            self._propagation_tree[(dataset, conversation_id)] = graph
            size_over_time = pd.Series(document['size_over_time'], name='Size')
            size_over_time.index = pd.to_datetime(size_over_time.index)
            self._size_over_time[(dataset, conversation_id)] = size_over_time
            depth_over_time = pd.Series(document['depth_over_time'], name='Depth')
            depth_over_time.index = pd.to_datetime(depth_over_time.index)
            self._depth_over_time[(dataset, conversation_id)] = depth_over_time
            max_breadth_over_time = pd.Series(document['max_breadth_over_time'], name='Max Breadth')
            max_breadth_over_time.index = pd.to_datetime(max_breadth_over_time.index)
            self._max_breadth_over_time[(dataset, conversation_id)] = max_breadth_over_time
            structural_virality_over_time = pd.Series(document['structural_virality_over_time'],
                                                      name='Structural Virality')
            structural_virality_over_time.index = pd.to_datetime(structural_virality_over_time.index)
            self._structural_virality_over_time[(dataset, conversation_id)] = structural_virality_over_time

        client.close()


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
