import logging

import fire
from pyaml_env import parse_config

from figures.propagation import PropagationPlotFactory

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def prepopulate_propagation(config_file='dev_config.yaml'):
    logger.debug('Prepopulating propagation metrics and graphs...')
    config = parse_config(config_file)
    factory = PropagationPlotFactory(available_datasets=config['available_datasets'],
                                     host=config['mongodb']['host'],
                                     port=config['mongodb']['port'],
                                     threshold=config['graph_simplification']['threshold'])
    factory.prepopulate()
    logger.debug('Propagation metrics and graphs prepopulated')


if __name__ == '__main__':
    # fire.Fire(prepopulate_propagation)
    prepopulate_propagation('dev_config.yaml')
