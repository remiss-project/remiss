import logging

from pyaml_env import parse_config

from figures.propagation import PropagationPlotFactory
from propagation import Egonet, NetworkMetrics, DiffusionMetrics
from propagation.histogram import Histogram

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Prepopulator:
    def __init__(self, config_file='prod_config.yaml',
                 metrics=('propagation', 'diffusion', 'network', 'egonet', 'histogram')):
        self.metrics = metrics
        self.config_file = config_file
        config = parse_config(config_file)
        self.diffusion_metrics = DiffusionMetrics(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                  reference_types=config['reference_types'])
        self.network_metrics = NetworkMetrics(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                              reference_types=config['reference_types'])
        self.egonet = Egonet(host=config['mongodb']['host'], port=config['mongodb']['port'],
                             reference_types=config['reference_types'],
                             threshold=config['graph_simplification']['threshold'])
        self.histogram = Histogram(host=config['mongodb']['host'], port=config['mongodb']['port'])
        self.propagation_factory = PropagationPlotFactory(host=config['mongodb']['host'],
                                                          port=config['mongodb']['port'],
                                                          available_datasets=config['available_datasets'])
        self.available_datasets = config['available_datasets']

    def _execute_with_logging(self, metric_type, persist_method, available_datasets, error_message):
        logger.info(f'Generating {metric_type}')
        try:
            persist_method(available_datasets)
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            raise RuntimeError(f"{error_message}: {e}") from e

    def generate_propagation_factory_metrics(self, available_datasets):
        self._execute_with_logging(
            "propagation metrics and graphs",
            self.propagation_factory.persist,
            available_datasets,
            "Error generating propagation metrics and graphs"
        )

    def generate_diffusion_metrics(self, available_datasets):
        self._execute_with_logging(
            "diffusion metrics",
            self.diffusion_metrics.persist,
            available_datasets,
            "Error generating diffusion metrics"
        )

    def generate_network_metrics(self, available_datasets):
        self._execute_with_logging(
            "network metrics",
            self.network_metrics.persist,
            available_datasets,
            "Error generating network metrics"
        )

    def generate_egonet_metrics(self, available_datasets):
        self._execute_with_logging(
            "egonet metrics",
            self.egonet.persist,
            available_datasets,
            "Error generating egonet metrics"
        )

    def generate_histograms(self, available_datasets):
        self._execute_with_logging(
            "histograms",
            self.histogram.persist,
            available_datasets,
            "Error generating histograms"
        )

    def prepopulate(self):
        logger.debug(f'Prepopulating propagation metrics and graphs from {self.config_file}...')
        if 'propagation' in self.metrics:
            self.generate_propagation_factory_metrics(self.available_datasets)
        if 'diffusion' in self.metrics:
            self.generate_diffusion_metrics(self.available_datasets)
        if 'network' in self.metrics:
            self.generate_network_metrics(self.available_datasets)
        if 'egonet' in self.metrics:
            self.generate_egonet_metrics(self.available_datasets)
        if 'histogram' in self.metrics:
            self.generate_histograms(self.available_datasets)
        logger.debug('All metrics and graphs prepopulated')


if __name__ == '__main__':
    prepopulator = Prepopulator(metrics=('propagation',))
    prepopulator.prepopulate()
