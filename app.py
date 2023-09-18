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

# import data with mongoimport first
# $ mongoimport --db test_remiss --collection <collection> --file <dataset>.preprocessed.jsonl --drop

app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
server = app.server

pymongoarrow.monkey.patch_all()

client = MongoClient("localhost", 27017)
database = client.get_database("test_remiss")
available_datasets = database.list_collection_names()
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
        dbc.RadioItems(options=[{"label": x, "value": x} for x in available_datasets],
                       value='test_tweets',
                       inline=True,
                       id='radio-buttons-dataset')
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure={}, id='temporal-evolution')
        ], width=6),
    ]),

], fluid=True)


# Add controls to build the interaction
@callback(
    Output(component_id='temporal-evolution', component_property='figure'),
    Input(component_id='radio-buttons-dataset', component_property='value')
)
def update_graph(dataset_chosen):
    client = MongoClient("localhost", 27017)
    database = client.get_database("test_remiss")
    collection = database.get_collection(dataset_chosen)
    # get available fields
    # available_fields = collection.find_one()
    df = collection.aggregate_pandas_all(
        [
            {'$group': {'_id': {'$dayOfYear': '$created_at'},
                        'count': {'$sum': 1}}}
        ],
        schema=Schema({'_id': int, 'count': int})
    )
    fig = px.histogram(df, x='continent', y=dataset_chosen, histfunc='avg')
    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run_server(debug=True)
