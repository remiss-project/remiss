import os
import time

import dash
import dash_bootstrap_components as dbc

from components import RemissDashboard
from figures import TweetUserPlotFactory, EgonetPlotFactory

REMISS_MONGODB_HOST = os.environ.get('REMISS_MONGODB_HOST', 'localhost')
REMISS_MONGODB_PORT = int(os.environ.get('REMISS_MONGODB_PORT', 27017))
REMISS_MONGODB_DATABASE = os.environ.get('REMISS_MONGODB_DATABASE', 'remiss')
REMISS_CACHE_DIR = os.environ.get('REMISS_CACHE_DIR', None)
REMISS_GRAPH_LAYOUT = os.environ.get('REMISS_GRAPH_LAYOUT', 'auto')
REMISS_GRAPH_SIMPLIFICATION = os.environ.get('REMISS_GRAPH_SIMPLIFICATION', 'backbone')
REMISS_GRAPH_SIMPLIFICATION_THRESHOLD = float(os.environ.get('REMISS_GRAPH_SIMPLIFICATION_THRESHOLD', 0.4))
REMISS_AVAILABLE_DATASETS = os.environ.get('REMISS_AVAILABLE_DATASETS', None)


def main():
    print(f'Connecting to MongoDB at {REMISS_MONGODB_HOST}:{REMISS_MONGODB_PORT}...')
    print(f'Using database {REMISS_MONGODB_DATABASE}...')
    if REMISS_CACHE_DIR:
        print(f'Using cache directory {REMISS_CACHE_DIR}...')
    else:
        print('Not using cache...')

    if REMISS_GRAPH_SIMPLIFICATION:
        print(f'Using graph simplification method {REMISS_GRAPH_SIMPLIFICATION} with threshold '
              f'{REMISS_GRAPH_SIMPLIFICATION_THRESHOLD}...')

    if REMISS_AVAILABLE_DATASETS:
        print(f'Using available datasets {REMISS_AVAILABLE_DATASETS}...')
        available_datasets = REMISS_AVAILABLE_DATASETS.split(',')
    else:
        available_datasets = None

    print('Creating plot factories...')
    start_time = time.time()
    tweet_user_plot_factory = TweetUserPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                                   database=REMISS_MONGODB_DATABASE,
                                                   available_datasets=available_datasets)
    egonet_plot_factory = EgonetPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                            database=REMISS_MONGODB_DATABASE, cache_dir=REMISS_CACHE_DIR,
                                            layout=REMISS_GRAPH_LAYOUT, simplification=REMISS_GRAPH_SIMPLIFICATION,
                                            threshold=REMISS_GRAPH_SIMPLIFICATION_THRESHOLD,
                                            available_datasets=available_datasets)
    dashboard = RemissDashboard(tweet_user_plot_factory, egonet_plot_factory)
    print(f'Plot factories created in {time.time() - start_time} seconds.')
    print('Creating app...')
    start_time = time.time()
    app = dash.Dash(external_stylesheets=[dbc.themes.JOURNAL])
    app.layout = dashboard.layout()
    dashboard.callbacks(app)
    print(f'App created in {time.time() - start_time} seconds.')
    print('Running app...')
    app.run(debug=True)


# Run the app
if __name__ == '__main__':
    main()
