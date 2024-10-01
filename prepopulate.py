import logging

from pyaml_env import parse_config

from figures.control import ControlPlotFactory
from figures.propagation import PropagationPlotFactory
from propagation import Egonet, NetworkMetrics, DiffusionMetrics
from propagation.histogram import Histogram

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Prepopulator:
    def __init__(self, config_file='prod_config.yaml', available_datasets=None,
                 modules=('diffusion', 'network', 'egonet', 'layout', 'histogram'), max_cascades=None):
        self.max_cascades = max_cascades
        self.modules = modules
        self.config_file = config_file
        config = parse_config(config_file)
        self.egonet = Egonet(host=config['mongodb']['host'], port=config['mongodb']['port'],
                             reference_types=config['reference_types'],
                             )
        self.diffusion_metrics = DiffusionMetrics(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                  reference_types=config['reference_types'], egonet=self.egonet)
        self.network_metrics = NetworkMetrics(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                              reference_types=config['reference_types'])

        self.histogram = Histogram(host=config['mongodb']['host'], port=config['mongodb']['port'])
        self.propagation_factory = PropagationPlotFactory(host=config['mongodb']['host'],
                                                          port=config['mongodb']['port'],
                                                          available_datasets=config['available_datasets'],
                                                          threshold=config['graph_simplification']['threshold'])
        self.control_plot_factory = ControlPlotFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                       available_datasets=config['available_datasets'],
                                                       max_wordcloud_words=config['wordcloud']['max_words'])
        self.available_datasets = config['available_datasets'] if available_datasets is None else available_datasets

    def _execute_with_logging(self, metric_type, persist_method, available_datasets, error_message):
        logger.info(f'Generating {metric_type}')
        try:
            persist_method(available_datasets)
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            raise RuntimeError(f"{error_message}: {e}") from e

    def generate_layout(self):
        self._execute_with_logging(
            "hidden network layout",
            self.propagation_factory.persist,
            self.available_datasets,
            "Error generating propagation metrics and graphs"
        )

    def generate_diffusion_metrics(self):
        logger.info(f'Generating diffusion metrics')
        try:
            self.diffusion_metrics.persist_diffusion_metrics(self.available_datasets, self.max_cascades)
        except Exception as e:
            logger.error(f"Error generating diffusion metrics: {e}")
            raise RuntimeError(f"Error generating diffusion metrics: {e}") from e

    def generate_diffusion_static_plots(self):
        self._execute_with_logging(
            "diffusion static plots",
            self.diffusion_metrics.persist_diffusion_static_plots,
            self.available_datasets,
            "Error generating diffusion static plots"
        )

    def generate_network_metrics(self):
        self._execute_with_logging(
            "network metrics",
            self.network_metrics.persist,
            self.available_datasets,
            "Error generating network metrics"
        )

    def generate_egonet_metrics(self):
        self._execute_with_logging(
            "egonet metrics",
            self.egonet.persist,
            self.available_datasets,
            "Error generating egonet metrics"
        )

    def generate_histograms(self):
        self._execute_with_logging(
            "histograms",
            self.histogram.persist,
            self.available_datasets,
            "Error generating histograms"
        )

    def generate_wordcloud_hashtag_freqs(self):
        self._execute_with_logging(
            "wordcloud hashtag frequencies",
            self.control_plot_factory.persist,
            self.available_datasets,
            "Error generating wordcloud hashtag frequencies"
        )

    def prepopulate(self):
        logger.debug(f'Prepopulating {self.available_datasets} with {self.modules}')
        for module in self.modules:
            match module:
                case 'layout':
                    self.generate_layout()
                case 'diffusion':
                    self.generate_diffusion_metrics()
                case 'diffusion_static_plots':
                    self.generate_diffusion_static_plots()
                case 'network':
                    self.generate_network_metrics()
                case 'egonet':
                    self.generate_egonet_metrics()
                case 'histogram':
                    self.generate_histograms()
                case 'wordcloud':
                    self.generate_wordcloud_hashtag_freqs()
                case _:
                    raise ValueError(f"Invalid module: {module}")
        logger.debug('All metrics and graphs prepopulated')


def run_prepopulator(config_file='prod_config.yaml', available_datasets=None,
                     modules=('egonet', 'layout', 'diffusion', 'diffusion_static_plots', 'network', 'histogram'),
                     max_cascades=None):
    logger.info(f'Running prepopulator with config file: {config_file}')
    logger.info(f'Available datasets: {available_datasets}')
    logger.info(f'Modules: {modules}')
    prepopulator = Prepopulator(config_file=config_file, available_datasets=available_datasets, modules=modules,
                                max_cascades=max_cascades)
    prepopulator.prepopulate()


if __name__ == '__main__':
    fire.Fire(run_prepopulator)
    # run_prepopulator(modules=['diffusion'], config_file='dev_config.yaml', max_cascades=10)
