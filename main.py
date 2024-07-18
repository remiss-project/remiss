import logging

import fire
from pyaml_env import parse_config

from figures.propagation import PropagationPlotFactory

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def prepopulate_propagation(config_file='dev_config.yaml'):
    logger.info('Prepopulating propagation metrics and graphs...')
    config = parse_config(config_file)
    factory = PropagationPlotFactory(available_datasets=config['available_datasets'],
                                     host=config['mongodb']['host'],
                                     port=config['mongodb']['port'],
                                     load_from_mongodb=False)
    factory.prepopulate()
    logger.info('Propagation metrics and graphs prepopulated')


if __name__ == '__main__':
    # fire.Fire(prepopulate_propagation)
    prepopulate_propagation('dev_config.yaml')
