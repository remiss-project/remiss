import json
from pathlib import Path

import networkx as nx
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pymongoarrow.schema import Schema
from tqdm import tqdm
from twarc import ensure_flattened
import zipfile
import pymongoarrow.monkey
import networkx
import plotly.graph_objects as go

pymongoarrow.monkey.patch_all()


def preprocess_tweets(twitter_jsonl_zip, metadata_file=None):
    """
    Preprocess the tweets gathered by running twarc and store them as a single tweet per line in a jsonl file named
    <twitter_jsonl_zip>.preprocessed.jsonl.
    Also store the media related information in a separate file named <twitter_jsonl_zip>.media.jsonl.
    Finally, store the tweets in a format in which the dates are correctly imported as such can by mongodbimport
    in a file named <twitter_jsonl_zip>.mongodbimport.jsonl.
    Optionally, if metadata_file is provided, add the metadata containing whether a tweet comes from
    a usual suspect or a politician to the tweets.

    $ mongoimport --db test_remiss --collection test_tweets --file test.mongodbimport.jsonl


    :param twitter_jsonl_zip: Source tweets in jsonl format, compressed in a zip file,as outputted by twarc
    :param metadata_file: Additional Laia metadata in xlsx format
    :return:
    """
    twitter_jsonl_zip = Path(twitter_jsonl_zip)
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            num_lines = sum(1 for _ in infile)
    if metadata_file:
        raw_metadata = pd.read_excel(metadata_file, sheet_name=None)
        # usual_suspects = []
        # for sheet_name, df in raw_metadata.items():
        #     if 'USUAL SUSPECTS' in sheet_name:
        #         usual_suspects.append(df)
        # usual_suspects = pd.concat(usual_suspects)
        usual_suspects = raw_metadata['NOVA LLISTA USUAL SUSPECTS']
        usual_suspects = usual_suspects.dropna(axis=1, how='all')
        usual_suspects = usual_suspects[usual_suspects['XARXA SOCIAL'] == 'Twitter']
        usernames = usual_suspects['ENLLAÇ'].str.split('/').str[-1].str.split('?').str[0]
        usual_suspects = usual_suspects.set_index(usernames)

        if not usual_suspects.index.is_unique:
            print('WARNING: usual suspects usernames are not unique')
            print(usual_suspects[usual_suspects.index.duplicated(keep=False)].sort_index())
            usual_suspects = usual_suspects.drop_duplicates(keep='first')

        parties = raw_metadata['LLISTA POLÍTICS']
        parties = parties.dropna(axis=1, how='all')
        parties = parties[parties['ENLLAÇ TW'] != 'No en té']
        usernames = parties['ENLLAÇ TW'].str.split('/').str[-1].str.split('?').str[0]
        parties = parties.set_index(usernames)
        if not parties.index.is_unique:
            print('WARNING: parties usernames are not unique')
            print(parties[parties.index.duplicated(keep=False)].sort_index())
            parties = parties.drop_duplicates(keep='first')

    output_jsonl = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.preprocessed.jsonl')
    output_media_urls = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.media.jsonl')
    output_mongodbimport = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.mongodbimport.jsonl')
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            with open(output_jsonl, "w") as outfile:
                with open(output_media_urls, "w") as media_outfile:
                    with open(output_mongodbimport, "w") as mongodbimport_outfile:
                        for line in tqdm(infile, total=num_lines):
                            line = json.loads(line)
                            tweets = ensure_flattened(line)
                            for tweet in tweets:
                                # store separately the media in another file
                                if 'attachments' in tweet and 'media' in tweet['attachments']:
                                    media = {'id': tweet['id'],
                                             'media': tweet['attachments']['media']}
                                    media_outfile.write(json.dumps(media) + "\n")
                                if metadata_file:
                                    username = tweet['author']['username']
                                    remiss_metadata = {'is_usual_suspect': username in usual_suspects.index}
                                    if username in parties.index:
                                        remiss_metadata['party'] = parties.loc[username, 'PARTIT']
                                    else:
                                        remiss_metadata['party'] = None
                                    tweet['author']['remiss_metadata'] = remiss_metadata

                                outfile.write(json.dumps(tweet) + "\n")

                                # convert the timestamps to mongodbimport format
                                # "starttime": "2019-12-01 00:00:05.5640"
                                # to
                                # "starttime": {
                                #     "$date": "2019-12-01T00:00:05.5640Z"
                                # }
                                fix_timestamps(tweet)
                                mongodbimport_outfile.write(json.dumps(tweet) + '\n')


def fix_timestamps(tweet):
    date_fields = {'created_at', 'editable_until', 'retrieved_at'}
    for field, value in tweet.items():
        if field in date_fields:
            if isinstance(value, dict):
                if '$date' not in value:
                    raise ValueError(f'Unexpected format in timestamp field {field}: {value}')
            else:
                date = value.replace(' ', 'T')
                # if not date.endswith('Z'):
                #     date += 'Z'
                tweet[field] = {'$date': date}
        elif isinstance(value, dict):
            fix_timestamps(value)


def load_tweet_count_evolution(host, port, database, collection, start_date=None, end_date=None, hashtag=None,
                               unit='day', bin_size=1, ):
    client = MongoClient(host, port)
    database = client.get_database(database)

    try:
        database.validate_collection(collection)
    except OperationFailure as ex:
        if ex.details['code'] == 26:
            raise ValueError(f'Dataset {collection} does not exist') from ex
        else:
            raise ex

    collection = database.get_collection(collection)
    pipeline = [
        {'$group': {
            "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
            "count": {'$count': {}}}},
        {'$sort': {'_id': 1}}
    ]
    if hashtag:
        pipeline.insert(0, {'$match': {'entities.hashtags.tag': hashtag}})
    if end_date:
        pipeline.insert(0, {'$match': {'created_at': {'$lte': pd.to_datetime(end_date)}}})
    if start_date:
        pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})

    df = collection.aggregate_pandas_all(
        pipeline,
        schema=Schema({'_id': datetime, 'count': int})
    )
    df = df.rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time').squeeze()
    return df


def load_user_count_evolution(host, port, database, collection, start_date=None, end_date=None, hashtag=None,
                              unit='day', bin_size=1, ):
    client = MongoClient(host, port)
    database = client.get_database(database)

    try:
        database.validate_collection(collection)
    except OperationFailure as ex:
        if ex.details['code'] == 26:
            raise ValueError(f'Dataset {collection} does not exist') from ex
        else:
            raise ex

    collection = database.get_collection(collection)
    pipeline = [
        {'$group': {
            "_id": {"$dateTrunc": {'date': "$created_at", 'unit': unit, 'binSize': bin_size}},
            "users": {'$addToSet': "$author.username"}}
        },
        {'$project': {'count': {'$size': '$users'}}},
        {'$sort': {'_id': 1}}
    ]
    if hashtag:
        pipeline.insert(0, {'$match': {'entities.hashtags.tag': hashtag}})
    if end_date:
        pipeline.insert(0, {'$match': {'created_at': {'$lte': pd.to_datetime(end_date)}}})
    if start_date:
        pipeline.insert(0, {'$match': {'created_at': {'$gte': pd.to_datetime(start_date)}}})

    df = collection.aggregate_pandas_all(
        pipeline,
        schema=Schema({'_id': datetime, 'count': int})
    )
    df = df.rename(columns={'_id': 'Time', 'count': 'Count'}).set_index('Time').squeeze()
    return df


def compute_hidden_network(host, port, database, dataset, reference_types=('replied_to', 'quoted', 'retweeted')):
    """
    Computes the hidden graph, this is, the graph of users that have interacted with each other.
    :param host: host where the mongodb instance is running
    :param port: port where the mongodb instance is running
    :param database: database where the tweets are stored
    :param collection: collection within the database where the tweets are stored
    :return: a networkx graph with the users as nodes and the edges representing interactions between users
    """
    client = MongoClient(host, port)
    database = client.get_database(database)
    collection = database.get_collection(dataset)

    graph = networkx.DiGraph()
    for tweet in tqdm(collection.find()):
        if 'referenced_tweets' in tweet:
            source = tweet['author']
            for referenced_tweet in tweet['referenced_tweets']:
                if referenced_tweet['type'] in reference_types:
                    target = get_reference_target_user(referenced_tweet, collection)
                    if source['id'] != target['id']:
                        if not graph.has_node(source['id']):
                            graph.add_node(source['id'], **source)
                        if not graph.has_node(target['id']):
                            graph.add_node(target['id'], **target)
                        graph.add_edge(source['id'], target['id'])

                else:
                    print(f'Tweet {tweet["id"]} has an unknown reference type {referenced_tweet["type"]}')

    client.close()

    return graph


def get_reference_target_user(referenced_tweet, collection):
    if 'author' in referenced_tweet:
        target = referenced_tweet['author']
    else:
        referenced_tweet_id = referenced_tweet['id']
        referenced_tweet = collection.find_one({'id': referenced_tweet_id})
        if referenced_tweet:
            target = referenced_tweet['author']
        else:
            print(f'Referenced tweet {referenced_tweet_id} not found')
            target = {'id': referenced_tweet_id}
    return target


def plot_network(network, layout='fruchterman_reingold'):
    if layout == 'fruchterman_reingold':
        layout = nx.fruchterman_reingold_layout(network)
    else:
        raise ValueError(f'Unknown layout {layout}')

    edge_x = []
    edge_y = []
    for edge in network.edges():
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in network.nodes():
        x, y = layout[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(network.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        try:
            username = network.nodes[adjacencies[0]]['username']
        except KeyError:
            username = adjacencies[0]
        node_text.append(f'{username}: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig


def compute_neighbourhood(host, port, database, dataset, chosen_user=None, radius=2):
    neighbourhood = compute_hidden_network(host, port, database, dataset)
    if chosen_user:
        user_id = get_user_id(host, port, database, dataset, chosen_user)
        neighbourhood = nx.ego_graph(neighbourhood, user_id, radius)

    return neighbourhood


def get_user_id(host, port, database, dataset, username):
    client = MongoClient(host, port)
    database = client.get_database(database)
    collection = database.get_collection(dataset)
    tweet = collection.find_one({'author.username': username})
    if tweet:
        return tweet['author']['id']
    else:
        raise RuntimeError(f'User {username} not found')

