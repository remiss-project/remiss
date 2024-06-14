import datetime
import random
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pyarrow
import pymongo
import pymongoarrow
from igraph import Layout
from matplotlib import pyplot as plt
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()

set_config(transform_output="pandas")


class PropagationPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, cache_dir=None,
                 reference_types=('replied_to', 'quoted', 'retweeted'), layout='fruchterman_reingold',
                 simplification=None, threshold=0.2, delete_vertices=True, k_cores=4, frequency='1D',
                 available_datasets=None, small_size_multiplier=50, big_size_multiplier=10):
        super().__init__(host, port, available_datasets)
        self.big_size_multiplier = big_size_multiplier
        self.small_size_multiplier = small_size_multiplier
        self.frequency = frequency
        self.bin_size = int(frequency[:-1])
        pd_units = {'D': 'day', 'W': 'week', 'M': 'month', 'Y': 'year'}
        self.unit = pd_units[frequency[-1]]
        self.delete_vertices = delete_vertices
        self.threshold = threshold
        self.reference_types = reference_types
        self._hidden_networks = {}
        self._simplified_hidden_networks = {}
        self.layout = layout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.simplification = simplification
        self.k_cores = k_cores

    def get_conversation_sizes(self, dataset):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
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
            {'$sort': {'size': 1}},
            {'$project': {'_id': 0, 'conversation_id': '$_id', 'size': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]}
                          }}
        ]
        df = collection.aggregate_pandas_all(pipeline)
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
            {'$project': {'_id': 0, 'source': '$id', 'target': '$referenced_tweets.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target'}},

        ]

        references = collection.aggregate_pandas_all(references_pipeline)
        tweet_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$project': {'tweet_id': '$id',
                          'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
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
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party',
                          'created_at': '$referenced_tweets.created_at'
                          }}
        ]
        merge_tweets_and_references_pipeline = [
            {'$group': {'_id': '$tweet_id',
                        'author_id': {'$first': '$author_id'},
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'},
                        'created_at': {'$first': '$created_at'},
                        }
             },
            {'$project': {'_id': 0,
                          'tweet_id': '$_id',
                          'author_id': 1,
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]},
                          'created_at': 1
                          }
             }
        ]
        tweets_pipeline = [
            *referenced_tweet_pipeline,
            {'$unionWith': {'coll': 'raw', 'pipeline': tweet_pipeline}},  # Fetch missing tweets
            *merge_tweets_and_references_pipeline
        ]
        schema = Schema({'tweet_id': str, 'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str,
                         'created_at': datetime.datetime})
        tweets = collection.aggregate_pandas_all(tweets_pipeline, schema=schema)
        if conversation_id not in tweets['tweet_id'].values:
            # add dummy conversation id tweet picking data from the original tweet
            conversation_tweet = {'tweet_id': conversation_id,
                                  'author_id': '-',
                                  'username': '-',
                                  'is_usual_suspect': False,
                                  'party': None,
                                  'created_at': tweets['created_at'].min() - pd.Timedelta(1, 's')}

            # link it to the oldest tweet in the conversation
            oldest_tweet_id = tweets['tweet_id'].iloc[tweets['created_at'].idxmin()]
            references = pd.concat([pd.DataFrame([{'source': conversation_id, 'target': oldest_tweet_id}]),
                                    references], ignore_index=True)
            tweets = pd.concat([pd.DataFrame([conversation_tweet]), tweets], ignore_index=True)
        client.close()

        return conversation_id, tweets, references

    def get_propagation_tree(self, dataset, tweet_id):
        conversation_id, conversation_tweets, references = self.get_conversation(dataset, tweet_id)
        graph = self._get_graph(conversation_tweets, references)
        self.ensure_conversation_id(conversation_id, graph)

        return graph

    def _get_graph(self, vertices, edges):
        graph = ig.Graph(directed=True)

        graph.add_vertices(len(vertices))
        graph.vs['label'] = vertices['username'].tolist()
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
        fig = self.get_propagation_figure(graph)
        return fig

    def compute_layout(self, network):
        print(f'Computing {self.layout} layout')
        start_time = time.time()
        layout = network.layout(self.layout, dim=3)
        print(f'Layout computed in {time.time() - start_time} seconds')
        return layout

    def get_propagation_figure(self, network):
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

        color_map = {'normal': 'rgb(255, 234, 208)', 'suspect': 'rgb(247, 111, 142)',
                     'politician': 'rgb(111, 247, 142)',
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
                        f'Party: {party}<br>'
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

    @staticmethod
    def get_shortest_paths_to_conversation_id(graph):
        conversation_id_index = graph.vs.find(tweet_id=graph['conversation_id']).index
        shortest_paths = pd.Series(graph.as_undirected().shortest_paths_dijkstra(source=conversation_id_index)[0])
        shortest_paths = shortest_paths.replace(float('inf'), pd.NA)
        return shortest_paths

    def get_size_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        # get the difference between the first tweet and the rest in minutes
        size = pd.Series(graph.vs['created_at'], index=graph.vs['label'])
        size = size - graph.vs.find(tweet_id=graph['conversation_id'])['created_at']
        size = size.dt.total_seconds() / 60
        return size

    def plot_size_over_time(self, dataset, tweet_id):
        size = self.get_size_over_time(dataset, tweet_id)
        # temporal cumulative histogram over time. the x axis is in minutes
        fig = px.histogram(size, x=size, nbins=100, cumulative=True)
        # set the x axis to be in minutes
        fig.update_xaxes(title_text='Minutes')
        fig.update_yaxes(title_text='Cumulative number of tweets')

        return fig

    def get_depth_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        shortest_paths = self.get_shortest_paths_to_conversation_id(graph)
        created_at = pd.Series(graph.vs['created_at'])
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        depths = {}
        for i, time in enumerate(created_at):
            depths[time] = shortest_paths.iloc[:i + 1].max()

        depths = pd.Series(depths, name='Depth')
        return depths

    def plot_depth_over_time(self, dataset, tweet_id):
        depths = self.get_depth_over_time(dataset, tweet_id)
        fig = px.line(depths, x=depths.index, y=depths.values)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Depth')

        return fig

    def get_max_breadth_over_time(self, dataset, tweet_id):
        graph = self.get_propagation_tree(dataset, tweet_id)
        shortest_paths = self.get_shortest_paths_to_conversation_id(graph)
        created_at = pd.Series(graph.vs['created_at'])
        order = created_at.argsort()
        shortest_paths = shortest_paths.iloc[order]
        created_at = created_at.iloc[order]
        max_breadth = {}
        for i, time in enumerate(created_at):
            max_breadth[time] = shortest_paths.iloc[:i + 1].value_counts().max()

        max_breadth = pd.Series(max_breadth, name='Max Breadth')
        return max_breadth

    def plot_max_breadth_over_time(self, dataset, tweet_id):
        max_breadth = self.get_max_breadth_over_time(dataset, tweet_id)

        fig = px.line(max_breadth, x=max_breadth.index, y=max_breadth.values)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Max Breadth')

        return fig

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

    def get_structural_virality_over_time(self, dataset, tweet_id):
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

        structured_virality = pd.Series(structured_virality, name='Structured Virality')
        return structured_virality

    def plot_structural_virality_over_time(self, dataset, tweet_id):
        structured_virality = self.get_structural_virality_over_time(dataset, tweet_id)

        fig = px.line(structured_virality, x=structured_virality.index, y=structured_virality.values)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Structured Virality')

        return fig

    def get_full_graph(self, dataset):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.id': {'$exists': True},
                        'referenced_tweets.author': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'target': '$id', 'source': '$referenced_tweets.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target'}},
        ]
        edges = collection.aggregate_pandas_all(pipeline)
        nested_pipeline = [
            {'$project': {'tweet_id': '$id',
                          'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          'created_at': 1
                          }}
        ]
        tweet_ids_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.id': {'$exists': True},
                        'referenced_tweets.author': {'$exists': True},
                        'created_at': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'tweet_id': '$referenced_tweets.id',
                          'author_id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party',
                          'created_at': 1
                          }},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing tweets
            {'$group': {'_id': '$tweet_id',
                        'author_id': {'$first': '$author'},
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'},
                        'created_at': {'$first': '$created_at'},
                        }
             },
            {'$project': {'_id': 0,
                          'tweet_id': '$_id',
                          'author_id': 1,
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]},
                          'created_at': 1
                          }
             }
        ]
        schema = Schema({'tweet_id': str, 'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str,
                         'created_at': str})
        tweets = collection.aggregate_pandas_all(tweet_ids_pipeline, schema=schema)
        client.close()
        tweets['created_at'] = pd.to_datetime(tweets['created_at'])
        graph = self._get_graph(tweets, edges)
        return graph

    def plot_propagation(self, dataset):
        graph = self.get_full_graph(dataset)
        fig = self.get_propagation_figure(graph)
        return fig

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

    def plot_depth_cascade_ccdf(self, dataset):
        ccdf = self.get_depth_cascade_ccdf(dataset)
        fig = px.line(ccdf, x=ccdf.index, y=ccdf.columns, log_y=True)

        return fig

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

    def plot_size_cascade_ccdf(self, dataset):
        ccdf = self.get_size_cascade_ccdf(dataset)
        fig = px.line(ccdf, x=ccdf.index, y=ccdf.columns, log_y=True)

        return fig

    def get_cascade_count_over_time(self, dataset):
        conversation_ids = self.get_conversation_ids(dataset)
        conversation_ids['user_type'] = conversation_ids.apply(transform_user_type, axis=1)
        conversation_ids = conversation_ids.drop(columns=['is_usual_suspect', 'party'])
        conversation_ids = conversation_ids.rename(columns={'is_normal': 'Normal', 'is_suspect': 'Suspect',
                                                            'is_politician': 'Politician',
                                                            'is_suspect_politician': 'Suspect politician'})
        conversation_ids = conversation_ids.set_index('created_at')
        conversation_ids = conversation_ids.resample('M').count()
        conversation_ids = conversation_ids.fillna(0)
        return conversation_ids

    def plot_cascade_count_over_time(self, dataset):
        conversation_ids = self.get_cascade_count_over_time(dataset)
        fig = px.line(conversation_ids, x=conversation_ids.index, y=conversation_ids.columns, log_y=True,
                      title='Number of cascades over time', labels={'value': 'Number of cascades (log scale)'},
                      category_orders={'variable': ['Normal', 'Suspect', 'Politician', 'Suspect politician']}
                      )

        fig.update_yaxes(title_text='Number of cascades (log scale)')
        fig.update_xaxes(title_text='Time')

        return fig

    def get_conversation_ids(self, dataset):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # get tweets that are the start of the conversation, so its conversation_id is the tweet_id
        pipeline = [
            {'$match': {'$expr': {'$eq': ['$id', '$conversation_id']}}},
            {'$project': {'_id': 0, 'conversation_id': 1,
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          'created_at': 1
                          }}
        ]
        schema = Schema({'conversation_id': str, 'is_usual_suspect': bool, 'party': str,
                         'created_at': datetime.datetime})
        df = collection.aggregate_pandas_all(pipeline, schema=schema)
        client.close()
        return df

    def persist_propagation_metrics(self, dataset):
        # Get
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('conversation_propagation')
        collection.drop()
        # Get structural virality and time span per conversation
        structural_virality = self.get_structural_viralities(dataset)
        # Cast timespan to seconds
        structural_virality['timespan'] = structural_virality['timespan'].dt.total_seconds()
        # Persist
        collection.insert_many(structural_virality.reset_index().to_dict('records'))

        client.close()

    def load_propagation_metrics_from_db(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('conversation_propagation')
        data = collection.find()
        structural_virality = pd.DataFrame(data)
        # Cast seconds to timedelta
        structural_virality['timespan'] = pd.to_timedelta(structural_virality['timespan'], unit='s')
        client.close()
        return structural_virality[['conversation_id', 'structured_virality', 'timespan']].set_index('conversation_id')

    def prepopulate(self, force=False):
        self.prepopulate_propagation(force)
        self.prepopulate_egonet(force)

    def prepopulate_egonet(self, force=False):
        if not self.cache_dir:
            print('WARNING: Cache directory not set')

        for dataset in (pbar := tqdm(self.available_datasets, desc='Prepopulating egonet')):
            pbar.set_postfix_str(dataset)
            try:
                if force or not self.is_cached(dataset, 'hidden_network'):
                    network = self._compute_hidden_network(dataset)
                    if self.cache_dir:
                        stem = f'hidden_network'
                        self.save_to_cache(dataset, network, stem)
                else:
                    print(f'Hidden network for {dataset} already cached')
            except Exception as ex:
                print(f'Error prepopulating egonet for {dataset}: {ex}')

    def prepopulate_propagation(self, force=False):
        if not self.cache_dir:
            print('WARNING: Cache directory not set')

        for dataset in (pbar := tqdm(self.available_datasets, desc='Prepopulating propagation')):
            pbar.set_postfix_str(dataset)
            try:
                if not force:
                    try:
                        self.load_propagation_metrics_from_db(dataset)
                        print(f'{dataset} already prepopulated, skipping...')
                        continue
                    except pymongo.errors.PyMongoError as e:
                        pass

                self.persist_propagation_metrics(dataset)
            except Exception as e:
                print(f'Error prepopulating propagation metrics for {dataset}: {e}')

    def generate_propagation_dataset(self, dataset, negative_sample_ratio=0.1):
        client = MongoClient(self.host, self.port)
        self._validate_dataset(client, dataset)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        # get tweets that are the start of the conversation, so its conversation_id is the tweet_id
        pipeline = [
            {'$match': {'$expr': {'$eq': ['$id', '$conversation_id']}}},
            {'$project': {'_id': 0,
                          'author_id': '$author.id',
                          'conversation_id': 1,
                          'num_hashtags': {'$size': {'$ifNull': ['$entities.hashtags', []]}},
                          'num_mentions': {'$size': {'$ifNull': ['$entities.mentions', []]}},
                          'num_urls': {'$size': {'$ifNull': ['$entities.urls', []]}},
                          'num_media': {'$size': {'$ifNull': ['$entities.media', []]}},
                          'num_interactions': {'$size': {'$ifNull': ['$referenced_tweets', []]}},
                          'num_words': {'$size': {'$split': ['$text', ' ']}},
                          'num_chars': {'$strLenCP': '$text'},
                          'is_usual_suspect_op': '$author.remiss_metadata.is_usual_suspect',
                          'party_op': '$author.remiss_metadata.party',
                          'num_tweets_op': '$author.public_metrics.tweet_count',
                          'num_followers_op': '$author.public_metrics.followers_count',
                          'num_following_op': '$author.public_metrics.following_count',
                          }},
        ]
        tweet_features = collection.aggregate_pandas_all(pipeline)

        propagation_metrics_pipeline = [
            {'$project': {'_id': 0, 'author_id': 1, 'legitimacy': 1, 't-closeness': 1}}
        ]
        collection = database.get_collection('user_propagation')
        propagation_metrics = collection.aggregate_pandas_all(propagation_metrics_pipeline)

        collection = database.get_collection('raw')
        user_features_pipeline = [
            {'$project': {'_id': 0,
                          'author_id': '$author.id',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          'num_tweets': '$author.public_metrics.tweet_count',
                          'num_followers': '$author.public_metrics.followers_count',
                          'num_following': '$author.public_metrics.following_count', }}
        ]
        user_features = collection.aggregate_pandas_all(user_features_pipeline)
        user_features = user_features.drop_duplicates(subset='author_id')
        user_features = user_features.merge(propagation_metrics, on='author_id', how='left').set_index('author_id')

        collection = database.get_collection('raw')
        edge_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}
                        }},
            {'$project': {'_id': 0, 'source': '$referenced_tweets.author.id', 'target': '$author.id',
                          'conversation_id': '$conversation_id'}}
        ]
        edges = collection.aggregate_pandas_all(edge_pipeline)
        client.close()

        features = edges.merge(tweet_features, left_on='conversation_id', right_on='conversation_id', how='inner')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_prev'), left_on='source',
                                  right_index=True,
                                  how='inner')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_curr'), left_on='target',
                                  right_on='author_id', how='inner')
        # Drop superflous columns
        features = features.drop(columns=['conversation_id', 'source', 'target', 'author_id'])

        # get negatives: for each source, target and conversation id, find another target from a different conversation
        # that is not the source
        negatives = []
        for source, interactions in edges.groupby('source'):
            if len(interactions) > 1:
                targets = set(interactions['target'].unique())
                for conversation_id, conversation in interactions.groupby('conversation_id'):
                    other_targets = pd.DataFrame(targets - set(conversation['target']), columns=['target'])
                    if len(other_targets) > 0:
                        other_targets = other_targets.sample(n=min([len(conversation), len(other_targets)]))

                        other_targets['source'] = source
                        other_targets['conversation_id'] = conversation_id
                        negatives.append(other_targets)


        negatives = pd.concat(negatives)
        negatives = negatives.merge(tweet_features, left_on='conversation_id', right_on='conversation_id', how='inner')
        negatives = negatives.merge(user_features.rename(columns=lambda x: f'{x}_prev'), left_on='source',
                                    right_index=True,
                                    how='inner')
        negatives = negatives.merge(user_features.rename(columns=lambda x: f'{x}_curr'), left_on='target',
                                    right_on='author_id', how='inner')

        print('Features generated')
        print(f'Num positives: {len(features)}')
        print(f'Num negatives: {len(negatives)}')
        df = negatives.groupby('source').size().reset_index(name='count')
        print(f'Negatives per source: {df["count"].mean()}')
        df.plot.hist(title='Distribution of negatives per source', logy=True, bins=20)
        plt.show()

        negatives = negatives.drop(columns=['conversation_id', 'source', 'target', 'author_id'])

        features['propagated'] = 1
        negatives['propagated'] = 0
        features = pd.concat([features, negatives])

        return features

    def fit_propagation_model(self, dataset):
        print('Fitting propagation model')
        df = self.generate_propagation_dataset(dataset)
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
        return pipeline

    def plot_egonet(self, collection, user, depth, start_date=None, end_date=None):
        network = self.get_egonet(collection, user, depth)
        network = network.as_undirected(mode='collapse')

        return self.get_egonet_figure(network, start_date, end_date)

    def get_egonet(self, dataset, user, depth):
        """
        Returns the egonet of a user of a certain date and depth if present,
        otherwise returns the simplified hidden network
        :param dataset:
        :param user:
        :param depth:
        :return:
        """
        hidden_network = self.get_hidden_network(dataset)
        # check if the user is in the hidden network
        if user:
            try:
                node = hidden_network.vs.find(username=user)
                neighbours = hidden_network.neighborhood(node, order=depth)
                egonet = hidden_network.induced_subgraph(neighbours)
                return egonet
            except (RuntimeError, ValueError) as ex:
                print(f'Computing neighbourhood for user {user} failed with error {ex}')
        if self.simplification:
            return self.get_simplified_hidden_network(dataset)
        else:
            return hidden_network

    def get_hidden_network(self, dataset):
        stem = f'hidden_network'
        if dataset not in self._hidden_networks:
            if self.cache_dir and self.is_cached(dataset, stem):
                network = self.load_from_cache(dataset, stem)
            else:
                network = self._compute_hidden_network(dataset)
                layout = self.compute_layout(network)
                network['layout'] = layout
                if self.cache_dir:
                    self.save_to_cache(dataset, network, stem)
            self._hidden_networks[dataset] = network

        return self._hidden_networks[dataset]

    def get_simplified_hidden_network(self, dataset):
        stem = f'hidden_network-{self.simplification}-{self.threshold}'
        if dataset not in self._simplified_hidden_networks:
            if self.cache_dir and self.is_cached(dataset, stem):
                network = self.load_from_cache(dataset, stem)
            else:
                network = self.get_hidden_network(dataset)
                network = self._simplify_graph(network)
                if self.cache_dir:
                    self.save_to_cache(dataset, network, stem)
            self._simplified_hidden_networks[dataset] = network

        return self._simplified_hidden_networks[dataset]

    def is_cached(self, dataset, stem):
        if not self.cache_dir:
            return False
        dataset_dir = self.cache_dir / dataset
        hn_graph_file = dataset_dir / f'{stem}.graphmlz'
        return hn_graph_file.exists()

    def load_from_cache(self, dataset, stem):
        dataset_dir = self.cache_dir / dataset
        hn_graph_file = dataset_dir / f'{stem}.graphmlz'
        network = ig.read(hn_graph_file)
        return network

    def save_to_cache(self, dataset, network, stem):
        dataset_dir = self.cache_dir / dataset
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        hn_graph_file = dataset_dir / f'{stem}.graphmlz'
        network.write_graphmlz(str(hn_graph_file))

    def get_legitimacy(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': '$author.id',
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'legitimacy': 1}},
        ]
        print('Computing legitimacy')
        start_time = time.time()
        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        legitimacy = legitimacy.set_index('author_id')
        legitimacy = legitimacy.sort_values('legitimacy', ascending=False)
        print(f'Legitimacy computed in {time.time() - start_time} seconds')
        return legitimacy

    def get_t_closeness(self, graph):
        closeness = graph.closeness(mode='out', )
        closeness = pd.Series(closeness, index=graph.vs['author_id']).fillna(0)
        return closeness

    def _get_legitimacy_per_time(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$group': {'_id': {'author': '$author.id',
                                'date': {
                                    "$dateTrunc": {'date': "$created_at", 'unit': self.unit, 'binSize': self.bin_size}}
                                },
                        'legitimacy': {'$count': {}}}},
            {'$project': {'_id': 0,
                          'author_id': '$_id.author',
                          'date': '$_id.date',
                          'legitimacy': 1}},
        ]
        print('Computing reputation')

        legitimacy = collection.aggregate_pandas_all(node_pipeline)
        if len(legitimacy) == 0:
            raise ValueError(
                f'No data available for the selected time range and dataset: {dataset} {self.unit} {self.bin_size}')
        legitimacy = legitimacy.pivot(columns='date', index='author_id', values='legitimacy')
        legitimacy = legitimacy.fillna(0)
        return legitimacy

    def get_reputation(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_per_time(dataset)
        reputation = legitimacy.cumsum(axis=1)

        print(f'Reputation computed in {time.time() - start_time} seconds')
        return reputation

    def get_status(self, dataset):
        start_time = time.time()
        legitimacy = self._get_legitimacy_per_time(dataset)
        reputation = legitimacy.cumsum(axis=1)
        status = reputation.apply(lambda x: x.argsort())
        print(f'Status computed in {time.time() - start_time} seconds')
        return status

    def _add_date_filters(self, pipeline, start_date, end_date):
        if start_date:
            pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})
        if end_date:
            pipeline.insert(0, {'$match': {'created_at': {'$lt': pd.to_datetime(end_date)}}})

    def _get_authors(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')
        nested_pipeline = [
            {'$project': {'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          'num_tweets': '$author.public_metrics.tweet_count',
                          'num_followers': '$author.public_metrics.followers_count',
                          'num_following': '$author.public_metrics.following_count',
                          }
             }]
        self._add_date_filters(nested_pipeline, start_date, end_date)

        node_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'author_id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party',
                          'num_tweets': '$referenced_tweets.author.public_metrics.tweet_count',
                          'num_followers': '$referenced_tweets.author.public_metrics.followers_count',
                          'num_following': '$referenced_tweets.author.public_metrics.following_count'
                          }},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing authors
            {'$group': {'_id': '$author_id',
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'},
                        'num_tweets': {'$last': '$num_tweets'},
                        'num_followers': {'$last': '$num_followers'},
                        'num_following': {'$last': '$num_following'}

                        }},

            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]},
                          'num_tweets': 1,
                          'num_followers': 1,
                          'num_following': 1},
             }
        ]
        self._add_date_filters(node_pipeline, start_date, end_date)
        print('Computing authors')
        start_time = time.time()
        schema = Schema({'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str,
                         'num_tweets': int, 'num_followers': int, 'num_following': int})
        authors = collection.aggregate_pandas_all(node_pipeline, schema=schema)
        print(f'Authors computed in {time.time() - start_time} seconds')
        client.close()
        return authors

    def _get_references(self, dataset, start_date=None, end_date=None):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('raw')

        references_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'source': '$author.id', 'target': '$referenced_tweets.author.id'}},
            {'$group': {'_id': {'source': '$source', 'target': '$target'},
                        'weight': {'$count': {}}}},
            {'$project': {'_id': 0, 'source': '$_id.source', 'target': '$_id.target', 'weight': 1}},
            {'$group': {'_id': '$source',
                        'node_weight': {'$sum': '$weight'},
                        'references': {'$push': {'target': '$target', 'weight': '$weight'}}}},
            {'$unwind': '$references'},
            {'$project': {'_id': 0, 'source': '$_id', 'target': '$references.target',
                          'weight': '$references.weight',
                          'weight_inv': {'$divide': [1, '$references.weight']},
                          'weight_norm': {'$divide': ['$references.weight', '$node_weight']},
                          }},
        ]
        self._add_date_filters(references_pipeline, start_date, end_date)
        print('Computing references')
        start_time = time.time()
        references = collection.aggregate_pandas_all(references_pipeline)
        print(f'References computed in {time.time() - start_time} seconds')
        client.close()
        return references

    def _compute_hidden_network(self, dataset):
        """
        Computes the hidden graph, this is, the graph of users that have interacted with each other.
        :param dataset: collection name within the database where the tweets are stored
        :return: a networkx graph with the users as nodes and the edges representing interactions between users
        """
        authors = self._get_authors(dataset)
        references = self._get_references(dataset)

        if len(authors) == 0:
            # in case of no authors we return an empty graph
            return ig.Graph(directed=True)

        print('Computing graph')
        start_time = time.time()
        # switch id by position (which will be the node id in the graph) and set it as index
        author_to_id = authors['author_id'].reset_index().set_index('author_id')
        # convert references which are author id based to graph id based
        references['source'] = author_to_id.loc[references['source']].reset_index(drop=True)
        references['target'] = author_to_id.loc[references['target']].reset_index(drop=True)
        # we only have reputation and legitimacy for a subset of the authors, so the others will be set to nan
        available_legitimacy = self.get_legitimacy(dataset)
        available_reputation = self.get_reputation(dataset)
        available_status = self.get_status(dataset)
        reputation = pd.DataFrame(np.nan, index=author_to_id.index, columns=available_reputation.columns)
        reputation.loc[available_reputation.index] = available_reputation
        legitimacy = pd.Series(np.nan, index=author_to_id.index)
        legitimacy[available_legitimacy.index] = available_legitimacy['legitimacy']
        status = pd.DataFrame(np.nan, index=author_to_id.index, columns=available_status.columns)
        status.loc[available_status.index] = available_status

        g = ig.Graph(directed=True)
        g.add_vertices(len(authors))
        g.vs['author_id'] = authors['author_id']
        g.vs['username'] = authors['username']
        g.vs['is_usual_suspect'] = authors['is_usual_suspect']
        g.vs['party'] = authors['party']
        g['reputation'] = reputation
        g['status'] = status
        g.vs['legitimacy'] = legitimacy.to_list()
        available_t_closeness = self.get_t_closeness(g)
        g.vs['t-closeness'] = available_t_closeness.to_list()
        g.vs['num_connections'] = g.degree()
        g.vs['num_tweets'] = authors['num_tweets']
        g.vs['num_followers'] = authors['num_followers']
        g.vs['num_following'] = authors['num_following']

        g.add_edges(references[['source', 'target']].to_records(index=False).tolist())
        g.es['weight'] = references['weight']
        g.es['weight_inv'] = references['weight_inv']
        g.es['weight_norm'] = references['weight_norm']
        print(g.summary())
        print(f'Graph computed in {time.time() - start_time} seconds')

        self.persist_graph_metrics(dataset, g)

        return g

    def _simplify_graph(self, network):
        if self.simplification == 'maximum_spanning_tree':
            network = network.spanning_tree(weights=network.es['weight_inv'])
        elif self.simplification == 'k_core':
            network = network.k_core(self.k_cores)
        elif self.simplification == 'backbone':
            network = compute_backbone(network, self.threshold, self.delete_vertices)
        else:
            raise ValueError(f'Unknown simplification {self.simplification}')
        return network

    def get_egonet_figure(self, network, start_date=None, end_date=None):
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

        # Legitimacy -> vertex color
        # Reputation -> vertex size
        # Party / Usual suspect -> vertex marker

        metadata = pd.DataFrame({'is_usual_suspect': network.vs['is_usual_suspect'], 'party': network.vs['party']})

        marker_map = {(False, False): 'circle',
                      (False, True): 'square',
                      (True, False): 'diamond',
                      (True, True): 'cross'}

        markers = metadata.apply(lambda x: marker_map[(x['is_usual_suspect'], x['party'] is not None)], axis=1)

        if start_date:
            size = network['reputation'][start_date]
        else:
            size = network['reputation'].mean(axis=1)
        # Add 1 offset and set 1 as minimum size
        size = size + 1
        size = size.fillna(1)
        if len(network.vs) > 100:
            size = size / size.max() * self.small_size_multiplier
        else:
            size = size / size.max() * self.big_size_multiplier

        color = pd.Series(network.vs['legitimacy'])

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
            legitimacy_value = node["legitimacy"] if not np.isnan(node["legitimacy"]) else '-'
            reputation_value = network["reputation"].loc[node['author_id']]
            reputation_value = reputation_value[start_date] if start_date else reputation_value.mean()
            reputation_value = f'{reputation_value:.2f}' if not np.isnan(reputation_value) else '-'

            node_text = f'Username: {node["username"]}<br>' \
                        f'Is usual suspect: {is_usual_suspect}<br>' \
                        f'Party: {party}<br>' \
                        f'Legitimacy: {legitimacy_value}<br>' \
                        f'Reputation: {reputation_value}'
            text.append(node_text)

        node_trace = go.Scatter3d(x=layout['x'],
                                  y=layout['y'],
                                  z=layout['z'],
                                  mode='markers',
                                  marker=dict(symbol=markers,
                                              size=size,
                                              color=color,
                                              # coloscale set to $champagne: #ffead0ff;
                                              # to $bright-pink-crayola: #f76f8eff;
                                              colorscale=[[0, 'rgb(255, 234, 208)'], [1, 'rgb(247, 111, 142)']],
                                              colorbar=dict(thickness=20, title='Legitimacy'),
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

    def persist_graph_metrics(self, dataset, graph):
        # Get legitimacy, reputation, and status from the graph vertices
        legitimacy = pd.Series(graph.vs['legitimacy'], index=graph.vs['author_id'])
        t_closeness = pd.Series(graph.vs['t-closeness'], index=graph.vs['author_id'])
        reputation = graph['reputation']
        status = graph['status']
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('user_propagation')
        collection.drop()
        collection.insert_many([{'author_id': author_id,
                                 'legitimacy': legitimacy.get(author_id),
                                 't-closeness': t_closeness.get(author_id),
                                 'reputation': reputation.loc[author_id].to_json(date_format='iso'),
                                 'status': status.loc[author_id].to_json(date_format='iso')} for author_id in
                                graph.vs['author_id']])

    def load_graph_metrics_from_db(self, dataset):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        collection = database.get_collection('user_propagation')
        data = collection.find()
        legitimacy = {}
        reputation = {}
        status = {}
        for user in data:
            author_id = user['author_id']
            legitimacy[author_id] = user['legitimacy']
            # Load reputation as Series
            reputation[author_id] = pd.read_json(user['reputation'], typ='series')
            reputation[author_id].index = pd.to_datetime(reputation[author_id].index, unit='s')
            # Load status as Series
            status[author_id] = pd.read_json(user['status'], typ='series')
            status[author_id].index = pd.to_datetime(status[author_id].index, unit='s')

        legitimacy = pd.Series(legitimacy, name='legitimacy')
        reputation = pd.DataFrame(reputation).T
        reputation.index.name = 'author_id'
        reputation.columns.name = 'date'
        status = pd.DataFrame(status).T
        status.index.name = 'author_id'
        status.columns.name = 'date'
        return legitimacy, reputation, status


def compute_backbone(graph, alpha=0.05, delete_vertices=True):
    # Compute alpha for all edges (1 - weight_norm)^(degree_of_source_node - 1)
    weights = np.array(graph.es['weight_norm'])
    degrees = np.array([graph.degree(e[0]) for e in graph.get_edgelist()])
    alphas = (1 - weights) ** (degrees - 1)
    good = np.nonzero(alphas > alpha)[0]
    backbone = graph.subgraph_edges(graph.es.select(good), delete_vertices=delete_vertices)
    if 'layout' in graph.attributes():
        layout = pd.DataFrame(graph['layout'].coords, columns=['x', 'y', 'z'], index=graph.vs['author_id'])
        backbone['layout'] = Layout(layout.loc[backbone.vs['author_id']].values.tolist(), dim=3)
    return backbone


def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
