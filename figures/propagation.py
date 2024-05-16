import igraph as ig
import pandas as pd
import pymongoarrow
from pymongo import MongoClient

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()


class PropagationPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None, cache_dir=None,
                 reference_types=('replied_to', 'quoted', 'retweeted'), layout='fruchterman_reingold'):
        super().__init__(host, port, available_datasets)
        self.cache_dir = cache_dir
        self.reference_types = reference_types
        self.layout = layout

    def get_conversation_lengths(self, dataset):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # Compute the length of the conversation for conversation
        pipeline = [
            {'$match': {'conversation_id': {'$exists': True}}},
            {'$group': {'_id': '$conversation_id', 'length': {'$sum': 1}}},
            {'$sort': {'_id': 1}}
        ]
        df = collection.aggregate_pandas_all(pipeline)
        df = df.rename(columns={'_id': 'Conversation ID', 'length': 'Length'}).sort_values('Length', ascending=False)
        return df

    def get_propagation_tree(self, dataset, tweet_id):
        # get all tweets belonging to the conversation
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        references_pipeline = [
            {'$match': {'conversation_id': tweet_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.id': {'$exists': True}}},
            {'$project': {'_id': 0, 'target': '$id', 'source': '$referenced_tweets.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target'}},

        ]

        references = collection.aggregate_pandas_all(references_pipeline)
        nested_pipeline = [
            {'$match': {'conversation_id': tweet_id}},
            {'$project': {'tweet_id': '$id'}}
        ]
        tweet_ids_pipeline = [
            {'$match': {'conversation_id': tweet_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.id': {'$exists': True}}},
            {'$project': {'_id': 0, 'tweet_id': '$referenced_tweets.id'}},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing authors
            {'$group': {'_id': '$tweet_id'}},
            {'$project': {'_id': 0, 'tweet_id': '$_id'}}
        ]
        tweets = collection.aggregate_pandas_all(tweet_ids_pipeline)

        client.close()
        graph = self._get_graph(tweets, references)
        return graph

    def _get_graph(self, vertices, edges):
        # switch id by position (which will be the node id in the graph) and set it as index
        tweet_to_id = vertices['tweet_id'].reset_index().set_index('tweet_id')
        # convert references which are author id based to graph id based
        edges['source'] = tweet_to_id.loc[edges['source']].reset_index(drop=True)
        edges['target'] = tweet_to_id.loc[edges['target']].reset_index(drop=True)
        graph = ig.Graph(directed=True)
        graph.add_vertices(len(vertices))
        graph.add_edges(edges[['source', 'target']].to_records(index=False).tolist())
        graph.vs['label'] = vertices['tweet_id'].tolist()
        return graph

    def get_tweet(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        tweet = collection.find_one({'id': tweet_id})
        client.close()
        if tweet:
            return tweet
        else:
            raise RuntimeError(f'Tweet {tweet_id} not found')
