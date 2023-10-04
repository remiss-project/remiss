import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash_holoniq_wordcloud import DashWordcloud
import os

from dash import dcc, Dash, callback, Output, Input
from dash import html
from pymongo import MongoClient
import networkx as nx

import pymongoarrow.monkey
from pymongoarrow.api import Schema

from remiss import load_tweet_count_evolution, load_user_count_evolution, compute_hidden_network, plot_network, \
    compute_neighbourhood, get_available_users, get_available_hashtag_freqs, load_tweet_count_evolution_per_type, \
    load_user_count_evolution_by_type

REMISS_MONGODB_HOST = os.environ.get('REMISS_MONGODB_HOST', 'localhost')
REMISS_MONGODB_PORT = int(os.environ.get('REMISS_MONGODB_PORT', 27017))
REMISS_MONGODB_DATABASE = os.environ.get('REMISS_MONGODB_DATABASE', 'test_remiss')

app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
server = app.server

pymongoarrow.monkey.patch_all()

client = MongoClient(REMISS_MONGODB_HOST, REMISS_MONGODB_PORT)
database = client.get_database(REMISS_MONGODB_DATABASE)
available_datasets = database.list_collection_names()
default_dataset = database.get_collection(available_datasets[0])
min_date_allowed = default_dataset.find_one(sort=[('created_at', 1)])['created_at'].date()
max_date_allowed = default_dataset.find_one(sort=[('created_at', -1)])['created_at'].date()
available_parties = default_dataset.distinct('author.remiss_metadata.party')
available_parties = [str(x) for x in available_parties]
available_users = [str(x) for x in default_dataset.distinct('author.username')]
available_hashtags_freqs = list(default_dataset.aggregate([
    {'$unwind': '$entities.hashtags'},
    {'$group': {'_id': '$entities.hashtags.tag', 'count': {'$sum': 1}}},
    {'$sort': {'count': -1}}
]))
available_hashtags_freqs = [(x['_id'], x['count']) for x in available_hashtags_freqs]
client.close()

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('Remiss', className="text-primary text-center fs-3")
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(options=[{"label": x, "value": x} for x in available_datasets],
                         value=available_datasets[0],
                         id='dropdown-dataset'),
            dcc.DatePickerRange(
                id='temporal-evolution-date-picker-range',
                min_date_allowed=min_date_allowed,
                max_date_allowed=max_date_allowed,
                initial_visible_month=min_date_allowed,
                start_date=min_date_allowed,
                end_date=max_date_allowed,
                display_format='DD/MM/YYYY',
            ),
        ]),
        dbc.Col([
            dcc.Graph(figure={}, id='tweets-per-day')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            DashWordcloud(
                list=available_hashtags_freqs,
                width=600, height=400,
                rotateRatio=0.5,
                shrinkToFit=True,
                shape='circle',
                hover=True,
                id='wordcloud'),
        ]),
        dbc.Col([
            dcc.Graph(figure={}, id='users-per-day')
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='hashtag-evolution'),
            dcc.Dropdown(options=[{"label": x, "value": x} for x, _ in available_hashtags_freqs],
                         id='dropdown-hashtag'),

        ]),
        dbc.Col([
            dcc.Graph(figure={}, id='egonet'),
            dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users], id='dropdown-user'),
            dcc.Slider(min=1, max=5, step=1, value=2, id='range-slider'),
        ]),
    ]),

], fluid=True)


@callback(
    Output(component_id='temporal-evolution-date-picker-range', component_property='min_date_allowed'),
    Output(component_id='temporal-evolution-date-picker-range', component_property='max_date_allowed'),
    Output(component_id='temporal-evolution-date-picker-range', component_property='start_date'),
    Output(component_id='temporal-evolution-date-picker-range', component_property='end_date'),
    Output(component_id='wordcloud', component_property='list'),
    Input(component_id='dropdown-dataset', component_property='value')
)
def update_data_picker_range(chosen_dataset):
    available_hashtags_freqs = get_available_hashtag_freqs(REMISS_MONGODB_HOST,
                                                           REMISS_MONGODB_PORT,
                                                           REMISS_MONGODB_DATABASE,
                                                           chosen_dataset=chosen_dataset)
    return min_date_allowed, max_date_allowed, min_date_allowed, max_date_allowed, available_hashtags_freqs


# Add controls to build the interaction
@callback(
    Output(component_id='tweets-per-day', component_property='figure'),
    Output(component_id='users-per-day', component_property='figure'),
    Input(component_id='dropdown-dataset', component_property='value'),
    Input(component_id='temporal-evolution-date-picker-range', component_property='start_date'),
    Input(component_id='temporal-evolution-date-picker-range', component_property='end_date'),
    Input(component_id='wordcloud', component_property='click')
)
def update_graph(chosen_dataset, start_date, end_date, hashtag):
    hashtag = hashtag[0] if hashtag else None
    tweet_count = load_tweet_count_evolution_per_type(REMISS_MONGODB_HOST, REMISS_MONGODB_PORT, REMISS_MONGODB_DATABASE,
                                                      collection=chosen_dataset,
                                                      start_date=start_date,
                                                      end_date=end_date,
                                                      hashtag=hashtag)

    fig_tweets_per_day = px.area(tweet_count, labels={"value": "Count"})

    data_user_count = load_user_count_evolution_by_type(REMISS_MONGODB_HOST, REMISS_MONGODB_PORT,
                                                        REMISS_MONGODB_DATABASE,
                                                        collection=chosen_dataset,
                                                        start_date=start_date,
                                                        end_date=end_date,
                                                        hashtag=hashtag)
    fig_users_per_day = px.area(data_user_count, labels={"value": "Count"})
    return fig_tweets_per_day, fig_users_per_day


@callback(
    Output(component_id='egonet', component_property='figure'),
    Output(component_id='dropdown-user', component_property='options'),
    Input(component_id='dropdown-user', component_property='value'),
    Input(component_id='dropdown-dataset', component_property='value'),
    Input(component_id='range-slider', component_property='value')
)
def update_egonet(chosen_user, dataset, radius):
    neighbourhood = compute_neighbourhood(REMISS_MONGODB_HOST,
                                          REMISS_MONGODB_PORT,
                                          REMISS_MONGODB_DATABASE,
                                          dataset=dataset,
                                          chosen_user=chosen_user,
                                          radius=radius)

    fig = plot_network(neighbourhood)
    available_users = get_available_users(REMISS_MONGODB_HOST,
                                          REMISS_MONGODB_PORT,
                                          REMISS_MONGODB_DATABASE,
                                          dataset=dataset)
    return fig, available_users


@callback(
    Output(component_id='hashtag-evolution', component_property='figure'),
    Input(component_id='dropdown-hashtag', component_property='value'),
    Input(component_id='dropdown-dataset', component_property='value'),
    Input(component_id='temporal-evolution-date-picker-range', component_property='start_date'),
    Input(component_id='temporal-evolution-date-picker-range', component_property='end_date'),
)
def update_hashtag_evolution(chosen_hashtag, dataset, start_date, end_date):
    data = load_tweet_count_evolution_per_type(REMISS_MONGODB_HOST,
                                               REMISS_MONGODB_PORT,
                                               REMISS_MONGODB_DATABASE,
                                               collection=dataset,
                                               start_date=start_date,
                                               end_date=end_date,
                                               hashtag=chosen_hashtag)
    fig = px.area(data, labels={"value": "Count"})
    return fig


# Run the app


if __name__ == '__main__':
    app.run(debug=True)
