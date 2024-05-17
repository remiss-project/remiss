import time

import igraph as ig
import pandas as pd
import plotly.graph_objects as go
import pymongoarrow
from pymongo import MongoClient
from pymongoarrow.schema import Schema

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
                        'referenced_tweets.id': {'$exists': True},
                        'referenced_tweets.author': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'target': '$id', 'source': '$referenced_tweets.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target'}},

        ]

        references = collection.aggregate_pandas_all(references_pipeline)
        nested_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$project': {'tweet_id': '$id',
                          'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party'
                          }}
        ]
        tweet_ids_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.id': {'$exists': True},
                        'referenced_tweets.author': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'tweet_id': '$referenced_tweets.id',
                          'author_id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party'
                          }},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing tweets
            {'$group': {'_id': '$tweet_id',
                        'author_id': {'$first': '$author_id'},
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'}}},
            {'$project': {'_id': 0,
                          'tweet_id': '$_id',
                          'author_id': 1,
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]}}}
        ]
        schema = Schema({'tweet_id': str, 'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str})
        tweets = collection.aggregate_pandas_all(tweet_ids_pipeline, schema=schema)
        client.close()
        return tweets, references

    def get_propagation_tree(self, dataset, tweet_id):
        conversation_tweets, references = self.get_conversation(dataset, tweet_id)
        graph = self._get_graph(conversation_tweets, references)
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
        graph.vs['label'] = vertices['username'].tolist()
        graph.vs['tweet_id'] = vertices['tweet_id'].tolist()
        graph.vs['username'] = vertices['username'].tolist()
        graph.vs['author_id'] = vertices['author_id'].tolist()
        graph.vs['is_usual_suspect'] = vertices['is_usual_suspect'].tolist()
        graph.vs['party'] = vertices['party'].tolist()

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

    def plot_propagation_tree(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        fig = self.plot_network(graph)
        return fig

    def compute_layout(self, network):
        print(f'Computing {self.layout} layout')
        start_time = time.time()
        layout = network.layout(self.layout, dim=3)
        print(f'Layout computed in {time.time() - start_time} seconds')
        return layout

    def plot_network(self, network):
        if 'layout' not in network.attributes():
            layout = self.compute_layout(network)
        else:
            layout = network['layout']
        layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
        print('Computing plot for network')
        print(network.summary())
        start_time = time.time()
        edges = pd.DataFrame(network.get_edgelist(), columns=['source', 'target'])
        edge_positions = layout.iloc[edges.values.flatten()].reset_index(drop=True)
        nones = edge_positions[1::2].assign(x=None, y=None, z=None)
        edge_positions = pd.concat([edge_positions, nones]).sort_index().reset_index(drop=True)

        color_map = {'normal': 'rgb(255, 234, 208)', 'suspect': 'rgb(247, 111, 142)', 'politician': 'rgb(111, 247, 142)',
                     'suspect_politician': 'rgb(111, 142, 247)'}
        color = [color_map['normal'] if not is_usual_suspect else color_map[party] for is_usual_suspect, party in
                    zip(network.vs['is_usual_suspect'], network.vs['party'])]
        size = [3 if not is_usual_suspect else 10 for is_usual_suspect in network.vs['is_usual_suspect']]

        # metadata = pd.DataFrame({'is_usual_suspect': network.vs['is_usual_suspect'], 'party': network.vs['party']})

        edge_trace = go.Scatter3d(x=edge_positions['x'],
                                  y=edge_positions['y'],
                                  z=edge_positions['z'],
                                  mode='lines',
                                  line=dict(color='rgb(125,125,125)', width=1),
                                  hoverinfo='none',
                                  name='Interactions',
                                  showlegend=False
                                  )

        text = []
        for node in network.vs:
            is_usual_suspect = 'Yes' if node['is_usual_suspect'] else 'No'
            party = f'Party: {node["party"]}' if node['party'] else '-'

            node_text = f'Username: {node["username"]}<br>' \
                        f'Is usual suspect: {is_usual_suspect}<br>' \
                        f'Party: {party}<br>' \

            text.append(node_text)

        node_trace = go.Scatter3d(x=layout['x'],
                                  y=layout['y'],
                                  z=layout['z'],
                                  mode='markers',
                                  marker=dict(
                                      # symbol=markers,
                                      size=size,
                                      color=color,
                                      # coloscale set to $champagne: #ffead0ff;
                                      # to $bright-pink-crayola: #f76f8eff;
                                      # colorscale=[[0, 'rgb(255, 234, 208)'], [1, 'rgb(247, 111, 142)']],
                                      # colorbar=dict(thickness=20, title='Legitimacy'),
                                      line=dict(color='rgb(50,50,50)', width=0.5),
                                  ),
                                  text=text,
                                  hovertemplate='%{text}',
                                  name='',
                                  )

        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )

        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            # margin=dict(
            #     t=100
            # ),
            hovermode='closest',

        )

        data = [edge_trace, node_trace]
        fig = go.Figure(data=data, layout=layout)

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        fig.update_layout(scene_camera=camera)
        print(f'Plot computed in {time.time() - start_time} seconds')
        return fig
