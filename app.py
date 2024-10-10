import logging
import time

import dash
import dash_bootstrap_components as dbc
import fire
from dash_bootstrap_templates import load_figure_template
from pyaml_env import parse_config

from components import RemissDashboard
from figures import TimeSeriesFactory, TweetTableFactory
from figures.control import ControlPlotFactory
from figures.multimodal import MultimodalPlotFactory
from figures.profiling import ProfilingPlotFactory
from figures.propagation import PropagationPlotFactory
from figures.textual import TextualFactory

logger = logging.getLogger('app')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

banner = \
    '''
     ______     ______     __    __     __     ______     ______    
    /\  == \   /\  ___\   /\ "-./  \   /\ \   /\  ___\   /\  ___\   
    \ \  __<   \ \  __\   \ \ \-./\ \  \ \ \  \ \___  \  \ \___  \  
     \ \_\ \_\  \ \_____\  \ \_\ \ \_\  \ \_\  \/\_____\  \/\_____\ 
      \/_/ /_/   \/_____/   \/_/  \/_/   \/_/   \/_____/   \/_____/ 
    '''

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


def create_app(config):
    load_figure_template(config['theme'])

    logger.info(f'Connecting to MongoDB at {config["mongodb"]["host"]}:{config["mongodb"]["port"]}...')

    logger.info('Creating plot factories...')
    start_time = time.time()
    control_plot_factory = ControlPlotFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                              available_datasets=config['available_datasets'],
                                              max_wordcloud_words=config['wordcloud']['max_words'])
    tweet_user_plot_factory = TimeSeriesFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                available_datasets=config['available_datasets'])
    tweet_table_factory = TweetTableFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                            available_datasets=config['available_datasets'])
    propagation_config = config['propagation']
    propagation_factory = PropagationPlotFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                 layout=propagation_config['graph_layout'],
                                                 threshold=propagation_config.get('threshold', 0.2),
                                                 frequency=propagation_config['frequency'],
                                                 available_datasets=config['available_datasets'],
                                                 max_edges_propagation_tree=propagation_config['max_edges'].get(
                                                     'propagation_tree', None),
                                                 max_edges_hidden_network=propagation_config['max_edges'].get(
                                                     'hidden_network', None),
                                                 )
    textual_factory = TextualFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                     available_datasets=config['available_datasets'])
    profiling_factory = ProfilingPlotFactory(data_dir=config['profiling']['data_dir'],
                                             available_datasets=config['available_datasets'],
                                             host=config['mongodb']['host'], port=config['mongodb']['port'])
    multimodal_factory = MultimodalPlotFactory(data_dir=config['multimodal']['data_dir'],
                                               available_datasets=config['available_datasets'],
                                               host=config['mongodb']['host'], port=config['mongodb']['port'])

    dashboard = RemissDashboard(
        control_plot_factory,
        tweet_user_plot_factory,
        tweet_table_factory,
        propagation_factory,
        textual_factory,
        profiling_factory,
        multimodal_factory,
        wordcloud_width=config['wordcloud']['width'],
        wordcloud_height=config['wordcloud']['height'],
        match_wordcloud_width=config['wordcloud']['match_width'],
        target_api_url=config.get('target_api_url', 'http://localhost:5000/process_dataset'),
        name='dashboard',
        debug=config['debug'],
        page_size=config['tweet_table']['page_size'],
    )
    logger.info(f'Plot factories created in {time.time() - start_time} seconds.')
    logger.info('Creating app...')
    start_time = time.time()
    dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
    app = dash.Dash(__name__,
                    external_stylesheets=[available_theme_css[config['theme']], dbc.icons.FONT_AWESOME, dbc_css],
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
    logger.info(f'App created in {time.time() - start_time} seconds.')

    return app


def load_config(config):
    config = parse_config(config)
    return config


def main(config='dev_config.yaml'):
    logger.info(banner)
    logger.info(f'Loading config from {config}...')
    config = load_config(config)
    app = create_app(config)
    app.run(debug=config['debug'])


# Run the app
if __name__ == '__main__':
    fire.Fire(main)
