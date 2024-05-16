import igraph as ig
import pandas as pd
from pymongo import MongoClient

from figures.figures import MongoPlotFactory


class PropagationPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None, cache_dir=None,
                 reference_types=('replied_to', 'quoted', 'retweeted'), layout='fruchterman_reingold'):
        super().__init__(host, port, available_datasets)
        self.cache_dir = cache_dir
        self.reference_types = reference_types
        self.layout = layout

    def get_propagation_tree(self, dataset, tweet_id):
        tweet = self.get_tweet(dataset, tweet_id)
        # get the original tweet by getting the conversation_id
        conversation_id = tweet['conversation_id']
        original_tweet = self.get_tweet(dataset, conversation_id)
        # Build the graph starting from the original tweet
        vertices = []
        edges = []
        self._build_tree(dataset, original_tweet, vertices, edges)
        graph = self._get_graph(vertices, edges)
        return graph

    def _build_tree(self, dataset, tweet, vertices, edges):
        # Add the tweet to the vertices
        vertices.append(tweet['id'])
        # Add the edges to the graph
        for reference in tweet['referenced_tweets']:
            if reference['type'] in self.reference_types:
                edges.append((tweet['id'], reference['id']))
                # Get the referenced tweet
                try:
                    referenced_tweet = self.get_tweet(dataset, reference['id'])
                    self._build_tree(dataset, referenced_tweet, vertices, edges)
                except RuntimeError:
                    # If the tweet is not found we add it as vertice and skip the edges
                    vertices.append(reference['id'])


    def _get_graph(self, vertices, edges):
        vertices = pd.DataFrame(vertices, columns=['tweet_id'])
        edges = pd.DataFrame(edges, columns=['source', 'target'])
        # switch id by position (which will be the node id in the graph) and set it as index
        tweet_to_id = vertices['tweet_id'].reset_index().set_index('tweet_id')
        # convert references which are author id based to graph id based
        edges['source'] = tweet_to_id.loc[edges['source']].reset_index(drop=True)
        edges['target'] = tweet_to_id.loc[edges['target']].reset_index(drop=True)
        graph = ig.Graph(directed=True)
        graph.add_vertices(len(vertices))
        graph.add_edges(edges[['source', 'target']].to_records(index=False).tolist())
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
