import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

from dash import dcc
from dash import html
from pymongo import MongoClient
import pymongoarrow.monkey
from pymongoarrow.api import Schema

app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
server = app.server

pymongoarrow.monkey.patch_all()

client = MongoClient("localhost", 27017)
database = client.get_database("test_remiss")
collection = database.get_collection("test_tweets")

colors = {"background": "#FFFFFF", "text": "#1DA1F2"}

pymongoarrow.monkey.patch_all()

client = MongoClient("localhost", 27017)
database = client.get_database("test_remiss")
collection = database.get_collection("test_tweets")
boundaries = list(range(0, 10000, 100))
df = collection.aggregate_pandas_all([
    {'$project': {
        'retweet_count': '$public_metrics.retweet_count',
    }},
    {'$bucket': {
        'groupBy': '$retweet_count',
        'boundaries': boundaries,
        'default': 'Other',
        'output': {
            'count': {'$sum': 1},
        }
    }
    }
],
    schema=Schema({'_id': int, 'count': int})
)
# df['bins'] = boundaries[1:]
fig = px.bar(df, x='_id', y="count")

fig.update_layout(
    plot_bgcolor=colors["background"],
    paper_bgcolor=colors["background"],
    font_color=colors["text"],
)

app.layout = html.Div(
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Histogram of Retweets",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="An example using Dash and MongoDB to display the retweet histogram.",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        dcc.Graph(id="example-twitter", figure=fig),
    ],
)

if __name__ == "__main__":
    app.run_server(debug=True)
