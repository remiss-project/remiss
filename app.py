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
REMISS_GRAPH_LAYOUT = os.environ.get('REMISS_GRAPH_LAYOUT', 'drl_3d')


def main():
    print(f'Connecting to MongoDB at {REMISS_MONGODB_HOST}:{REMISS_MONGODB_PORT}...')
    print(f'Using database {REMISS_MONGODB_DATABASE}...')
    if REMISS_CACHE_DIR:
        print(f'Using cache directory {REMISS_CACHE_DIR}...')
    else:
        print('Not using cache...')

    print('Creating plot factories...')
    start_time = time.time()
    tweet_user_plot_factory = TweetUserPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                                   database=REMISS_MONGODB_DATABASE)
    egonet_plot_factory = EgonetPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                            database=REMISS_MONGODB_DATABASE, cache_dir=REMISS_CACHE_DIR,
                                            layout=REMISS_GRAPH_LAYOUT)
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
