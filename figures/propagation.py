import datetime
import random
import time

import igraph as ig
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import pyarrow
import pymongo
import pymongoarrow
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from tqdm import tqdm

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()


class PropagationPlotFactory(MongoPlotFactory):
    def __init__(self, host="localhost", port=27017, available_datasets=None, cache_dir=None,
                 reference_types=('replied_to', 'quoted', 'retweeted'), layout='fruchterman_reingold'):
        super().__init__(host, port, available_datasets)
        self.cache_dir = cache_dir
        self.reference_types = reference_types
        self.layout = layout

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
        graph = self._get_graph(conversation_id, conversation_tweets, references)

        return graph

    def _get_graph(self, conversation_id, vertices, edges):
        graph = ig.Graph(directed=True)
        graph['conversation_id'] = conversation_id

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

        # link connected components to the conversation id vertex
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
        graph = self._get_graph(None, tweets, edges)
        return graph

    def plot_propagation(self, dataset):
        graph = self.get_full_graph(dataset)
        fig = self.plot_network(graph)
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

    def prepare_propagation_dataset(self, dataset, num_negative_samples=0.1):
        """
        Prepare the dataset for propagation analysis by adding a 'retweeted_status' field to the tweets that are retweets
        and a 'quoted_status' field to the tweets that are quotes.

        It contains for each tweet and user the following fields for each user and tweet:

        user_features = ['is_usual_suspect', 'is_politician', 'legitimacy', 't-closeness', 'num_connections', 'num_tweets',
                            'num_followers', 'num_following']
        tweet_feature = ['num_hashtags', 'num_mentions', 'num_urls', 'num_media', 'num_interactions', 'num_replies',
                             'num_words', 'num_chars', 'num_emojis']

        Each sample consists of:

        - user features for the op user of the tweet
        - tweet features for the tweet
        - user features for the user that retweeted the tweet before the present user

        :param dataset: Name of the mongodb database containing the tweets
        :return: X, y matrices ready to be used for propagation analysis
        """
        plot_factory = PropagationPlotFactory()

        X, y = [], []
        conversations = plot_factory.get_conversation_ids(dataset)
        for conversation_id in conversations:
            propagation_tree = plot_factory.get_propagation_tree(dataset, conversation_id)
            op_tweet = propagation_tree.vs.find(tweet_id=conversation_id)
            op_tweet_features = self.get_tweet_features(dataset, conversation_id)
            op_author = op_tweet['author_id']
            op_author_feature = self.get_author_features(dataset, op_author)
            for tweet in propagation_tree.vs:
                if tweet['tweet_id'] != conversation_id:
                    previous_user = tweet['author_id']
                    previous_user_features = self.get_author_features(dataset, previous_user)
                    for next_user in propagation_tree.successors(tweet):
                        next_user_features = self.get_author_features(dataset, next_user['author_id'])
                        # Create sample
                        sample = [*op_author_feature, *op_tweet_features, *previous_user_features, *next_user_features]
                        X.append(sample)
                        # Add label
                        y.append(1)

                    # Add negative samples




def transform_user_type(x):
    if x['is_usual_suspect'] and x['party'] is not None:
        return 'Suspect politician'
    elif x['is_usual_suspect']:
        return 'Suspect'
    elif x['party'] is not None:
        return 'Politician'
    else:
        return 'Normal'
