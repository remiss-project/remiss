import logging

import fire
from pyaml_env import parse_config

from figures.propagation import PropagationPlotFactory
from propagation import Egonet, NetworkMetrics, DiffusionMetrics
from propagation.histogram import Histogram

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def prepopulate_propagation(config_file='dev_config.yaml'):
    logger.debug(f'Prepopulating propagation metrics and graphs from {config_file}...')
    config = parse_config(config_file)
    diffusion_metrics = DiffusionMetrics(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                         reference_types=config['reference_types'])
    network_metrics = NetworkMetrics(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                     reference_types=config['reference_types'])
    egonet = Egonet(host=config['mongodb']['host'], port=config['mongodb']['port'],
                    reference_types=config['reference_types'], threshold=config['graph_simplification']['threshold'])
    histogram = Histogram(host=config['mongodb']['host'], port=config['mongodb']['port'])
    propagation_factory = PropagationPlotFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                    available_datasets=config['available_datasets'])
    available_datasets = config['available_datasets']
    logger.info(f"Prepopulating datasets: {available_datasets}")

    logger.info('Generating propagation metrics and graphs')
    try:
        propagation_factory.persist(available_datasets)
    except Exception as e:
        logger.error(f"Error generating propagation metrics and graphs: {e}")
        raise RuntimeError(f"Error generating propagation metrics and graphs: {e}") from e

    logger.info('Generating diffusion metrics')
    try:
        diffusion_metrics.persist(available_datasets)
    except Exception as e:
        logger.error(f"Error generating diffusion metrics: {e}")
        raise RuntimeError(f"Error generating diffusion metrics: {e}") from e
    logger.info('Generating network metrics')
    try:
        network_metrics.persist(available_datasets)
    except Exception as e:
        logger.error(f"Error generating network metrics: {e}")
        raise RuntimeError(f"Error generating network metrics: {e}") from e
    logger.info('Generating egonet metrics')
    try:
        egonet.persist(available_datasets)
    except Exception as e:
        logger.error(f"Error generating egonet metrics: {e}")
        raise RuntimeError(f"Error generating egonet metrics: {e}") from e
    logger.info('Generating histograms')
    try:
        histogram.persist(available_datasets)
    except Exception as e:
        logger.error(f"Error generating histograms: {e}")
        raise RuntimeError(f"Error generating histograms: {e}") from e
    logger.debug('Propagation metrics and graphs prepopulated')


if __name__ == '__main__':
    fire.Fire(prepopulate_propagation)

