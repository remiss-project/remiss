import os
import time

import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

from components import RemissDashboard
from figures import TweetUserPlotFactory, EgonetPlotFactory, TopTableFactory

REMISS_MONGODB_HOST = os.environ.get('REMISS_MONGODB_HOST', 'localhost')
REMISS_MONGODB_PORT = int(os.environ.get('REMISS_MONGODB_PORT', 27017))
REMISS_MONGODB_DATABASE = os.environ.get('REMISS_MONGODB_DATABASE', 'remiss')
REMISS_CACHE_DIR = os.environ.get('REMISS_CACHE_DIR', None)
REMISS_GRAPH_LAYOUT = os.environ.get('REMISS_GRAPH_LAYOUT', 'auto')
REMISS_GRAPH_SIMPLIFICATION = os.environ.get('REMISS_GRAPH_SIMPLIFICATION', 'backbone')
REMISS_GRAPH_SIMPLIFICATION_THRESHOLD = float(os.environ.get('REMISS_GRAPH_SIMPLIFICATION_THRESHOLD', 0.95))
REMISS_AVAILABLE_DATASETS = os.environ.get('REMISS_AVAILABLE_DATASETS', None)
if REMISS_AVAILABLE_DATASETS:
    print(f'Using available datasets {REMISS_AVAILABLE_DATASETS}...')
    REMISS_AVAILABLE_DATASETS = REMISS_AVAILABLE_DATASETS.split(',')

REMISS_THEME = os.environ.get('REMISS_THEME', 'pulse').upper()
REMISS_DEBUG = os.environ.get('REMISS_DEBUG', 'False')
REMISS_DEBUG = REMISS_DEBUG.lower() == 'true'
REMISS_FREQUENCY = os.environ.get('REMISS_FREQUENCY', '1D')
REMISS_PREPOPULATE = os.environ.get('REMISS_PREPOPULATE', 'False')
REMISS_PREPOPULATE = REMISS_PREPOPULATE.lower() == 'true'

available_theme_css = {'BOOTSTRAP': dbc.themes.BOOTSTRAP,
                       'CERULEAN': dbc.themes.CERULEAN,
                       'COSMO': dbc.themes.COSMO,
                       'CYBORG': dbc.themes.CYBORG,
                       'DARKLY': dbc.themes.DARKLY,
                       'FLATLY': dbc.themes.FLATLY,
                       'JOURNAL': dbc.themes.JOURNAL,
                       'LITERA': dbc.themes.LITERA,
                       'LUMEN': dbc.themes.LUMEN,
                       'LUX': dbc.themes.LUX,
                       'MATERIA': dbc.themes.MATERIA,
                       'MINTY': dbc.themes.MINTY,
                       'MORPH': dbc.themes.MORPH,
                       'PULSE': dbc.themes.PULSE,
                       'QUARTZ': dbc.themes.QUARTZ,
                       'SANDSTONE': dbc.themes.SANDSTONE,
                       'SIMPLEX': dbc.themes.SIMPLEX,
                       'SKETCHY': dbc.themes.SKETCHY,
                       'SLATE': dbc.themes.SLATE,
                       'SOLAR': dbc.themes.SOLAR,
                       'SPACELAB': dbc.themes.SPACELAB,
                       'SUPERHERO': dbc.themes.SUPERHERO,
                       'UNITED': dbc.themes.UNITED,
                       'VAPOR': dbc.themes.VAPOR,
                       'YETI': dbc.themes.YETI,
                       'ZEPHYR': dbc.themes.ZEPHYR
                       }


def prepopulate():
    if REMISS_AVAILABLE_DATASETS:
        print(f'Using available datasets {REMISS_AVAILABLE_DATASETS}...')
        available_datasets = REMISS_AVAILABLE_DATASETS.split(',')
    else:
        available_datasets = None
    egonet_plot_factory = EgonetPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                            database=REMISS_MONGODB_DATABASE, cache_dir=REMISS_CACHE_DIR,
                                            layout=REMISS_GRAPH_LAYOUT, simplification=REMISS_GRAPH_SIMPLIFICATION,
                                            threshold=REMISS_GRAPH_SIMPLIFICATION_THRESHOLD,
                                            frequency=REMISS_FREQUENCY,
                                            available_datasets=REMISS_AVAILABLE_DATASETS,
                                            prepopulate=False)
    print('Prepopulating...')
    start_time = time.time()
    egonet_plot_factory.prepopulate_cache()
    print(f'Prepopulated in {time.time() - start_time} seconds.')


def create_app():
    load_figure_template(REMISS_THEME)
    print(f'Connecting to MongoDB at {REMISS_MONGODB_HOST}:{REMISS_MONGODB_PORT}...')
    print(f'Using database {REMISS_MONGODB_DATABASE}...')
    if REMISS_CACHE_DIR:
        print(f'Using cache directory {REMISS_CACHE_DIR}...')
    else:
        print('Not using cache...')

    if REMISS_GRAPH_SIMPLIFICATION:
        print(f'Using graph simplification method {REMISS_GRAPH_SIMPLIFICATION} with threshold '
              f'{REMISS_GRAPH_SIMPLIFICATION_THRESHOLD}...')



    print('Creating plot factories...')
    start_time = time.time()
    tweet_user_plot_factory = TweetUserPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                                   database=REMISS_MONGODB_DATABASE,
                                                   available_datasets=REMISS_AVAILABLE_DATASETS)
    top_table_factory = TopTableFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                        database=REMISS_MONGODB_DATABASE,
                                        available_datasets=REMISS_AVAILABLE_DATASETS)

    egonet_plot_factory = EgonetPlotFactory(host=REMISS_MONGODB_HOST, port=REMISS_MONGODB_PORT,
                                            database=REMISS_MONGODB_DATABASE, cache_dir=REMISS_CACHE_DIR,
                                            layout=REMISS_GRAPH_LAYOUT, simplification=REMISS_GRAPH_SIMPLIFICATION,
                                            threshold=REMISS_GRAPH_SIMPLIFICATION_THRESHOLD,
                                            frequency=REMISS_FREQUENCY,
                                            available_datasets=REMISS_AVAILABLE_DATASETS,
                                            prepopulate=REMISS_PREPOPULATE)
    dashboard = RemissDashboard(tweet_user_plot_factory, top_table_factory, egonet_plot_factory, debug=REMISS_DEBUG)
    print(f'Plot factories created in {time.time() - start_time} seconds.')
    print('Creating app...')
    start_time = time.time()
    dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    app = dash.Dash(__name__,
                    external_stylesheets=[available_theme_css[REMISS_THEME], dbc.icons.FONT_AWESOME, dbc_css],
                    prevent_initial_callbacks="initial_duplicate",
                    meta_tags=[
                        {
                            "name": "viewport",
                            "content": "width=device-width, initial-scale=1, maximum-scale=1",
                        }
                    ],
                    )
    app.layout = dashboard.layout()
    dashboard.callbacks(app)
    print(f'App created in {time.time() - start_time} seconds.')

    return app


# Run the app
if __name__ == '__main__':
    app = create_app()
    print('Running app...')
    app.run(debug=REMISS_DEBUG)
