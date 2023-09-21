import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

from dash import dcc, Dash, callback, Output, Input
from dash import html
from pymongo import MongoClient
import pymongoarrow.monkey
from pymongoarrow.api import Schema

from remiss import load_tweet_count_evolution

REMISS_MONGODB_HOST = os.environ.get('REMISS_MONGODB_HOST', 'localhost')
REMISS_MONGODB_PORT = int(os.environ.get('REMISS_MONGODB_PORT', 27017))
REMISS_MONGODB_DATABASE = os.environ.get('REMISS_MONGODB_DATABASE', 'test_remiss')

app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
server = app.server

pymongoarrow.monkey.patch_all()

client = MongoClient(REMISS_MONGODB_HOST, REMISS_MONGODB_PORT)
database = client.get_database(REMISS_MONGODB_DATABASE)
available_datasets = database.list_collection_names()
min_date_allowed = database.get_collection(available_datasets[0]).find_one(sort=[('created_at', 1)])[
    'created_at'].date()
max_date_allowed = database.get_collection(available_datasets[0]).find_one(sort=[('created_at', -1)])[
    'created_at'].date()
available_parties = database.get_collection(available_datasets[0]).distinct('author.remiss_metadata.party')
available_parties = [str(x) for x in available_parties]
available_users = [str(x) for x in database.get_collection(available_datasets[0]).distinct('author.username')]
available_hashtags = [str(x) for x in database.get_collection(available_datasets[0]).distinct('hashtags')]
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
            dcc.Graph(figure={}, id='wordcloud'),
        ]),
        dbc.Col([
            dcc.Graph(figure={}, id='users-per-day')
        ]),
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='hashtag-evolution'),
            dcc.Dropdown(options=[{"label": x, "value": x} for x in available_hashtags], id='dropdown-hashtag'),

        ]),
        dbc.Col([
            dcc.Graph(figure={}, id='egonet'),
            dcc.Dropdown(options=[{"label": x, "value": x} for x in available_users], id='dropdown-user'),
        ]),
    ]),

], fluid=True)

# Add controls to build the interaction
# @callback(
#     Output(component_id='temporal-evolution', component_property='figure'),
#     Input(component_id='dropdown-dataset', component_property='value'),
#     Input(component_id='temporal-evolution-date-picker-range', component_property='start_date'),
#     Input(component_id='temporal-evolution-date-picker-range', component_property='end_date')
# )
# def update_graph(chosen_dataset, start_date, end_date):
#     data = load_tweet_count_evolution(REMISS_MONGODB_HOST, REMISS_MONGODB_PORT, REMISS_MONGODB_DATABASE,
#                                       chosen_dataset, start_date, end_date)
#     fig = px.bar(data, labels={"value": "Count"})
#     return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
