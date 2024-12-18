import logging

import fire
from pyaml_env import parse_config

from figures import TweetTableFactory
from figures.control import ControlPlotFactory
from figures.propagation import PropagationPlotFactory
from propagation import Egonet, NetworkMetrics, DiffusionMetrics
from propagation.histogram import Histogram

logger = logging.getLogger('main')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Prepopulator:
    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted', 'replied_to'),
                 graph_layout='fruchterman_reingold',
                 propagation_threshold=0.2, propagation_frequency='1D', max_edges_propagation_tree=None,
                 max_edges_hidden_network=4000, wordcloud_max_words=None, available_datasets=None,
                 erase_existing=True, max_cascades=None,
                 modules=('layout', 'diffusion', 'diffusion_static_plots', 'network', 'egonet', 'histogram',
                          'wordcloud', 'tweet_table')
                 ):
        # Initialization of the class attributes
        self.erase_existing = erase_existing
        self.max_cascades = max_cascades
        self.modules = modules
        self.available_datasets = available_datasets

        # Initialize the various components with the provided arguments
        self.egonet = Egonet(
            host=host,
            port=port,
            reference_types=reference_types
        )
        self.diffusion_metrics = DiffusionMetrics(
            host=host,
            port=port,
            reference_types=reference_types,
            egonet=self.egonet
        )
        self.network_metrics = NetworkMetrics(
            host=host,
            port=port,
            reference_types=reference_types
        )
        self.histogram = Histogram(
            host=host,
            port=port
        )

        self.propagation_factory = PropagationPlotFactory(
            host=host,
            port=port,
            layout=graph_layout,
            threshold=propagation_threshold,
            frequency=propagation_frequency,
            available_datasets=available_datasets,
            max_edges_propagation_tree=max_edges_propagation_tree,
            max_edges_hidden_network=max_edges_hidden_network,
            preload=False
        )

        self.control_plot_factory = ControlPlotFactory(
            host=host,
            port=port,
            available_datasets=available_datasets,
            max_wordcloud_words=wordcloud_max_words
        )

        self.tweet_table_factory = TweetTableFactory(
            host=host,
            port=port,
            available_datasets=available_datasets
        )

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
            self.diffusion_metrics.persist_diffusion_metrics(self.available_datasets,
                                                             max_cascades=self.max_cascades,
                                                             erase_existing=self.erase_existing)
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

    def generate_tweet_table(self):
        self._execute_with_logging(
            "tweet table",
            self.tweet_table_factory.persist,
            self.available_datasets,
            "Error generating tweet table"
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
                case 'tweet_table':
                    self.generate_tweet_table()
                case _:
                    raise ValueError(f"Invalid module: {module}")
        logger.debug('All metrics and graphs prepopulated')


def run_prepopulator(config_file='prod_config.yaml', available_datasets=None,
                     modules=('egonet', 'layout', 'diffusion', 'diffusion_static_plots', 'network', 'histogram',
                              'tweet_table'),
                     max_cascades=23, erase_existing=True):
    logger.info(f'Running prepopulator with config file: {config_file}')
    logger.info(f'Available datasets: {available_datasets}')
    logger.info(f'Modules: {modules}')
    logger.info(f'Max cascades: {max_cascades}, erase existing: {erase_existing}')
    config = parse_config(config_file)
    prepopulator = Prepopulator(host=config['mongodb']['host'],
                                port=config['mongodb']['port'],
                                reference_types=config['reference_types'],
                                graph_layout=config['propagation']['graph_layout'],
                                propagation_threshold=config['propagation'].get('threshold', 0.2),
                                propagation_frequency=config['propagation']['frequency'],
                                max_edges_propagation_tree=config['propagation']['max_edges'].get('propagation_tree'),
                                max_edges_hidden_network=config['propagation']['max_edges'].get('hidden_network'),
                                wordcloud_max_words=config['wordcloud']['max_words'],
                                available_datasets=config['available_datasets'],
                                modules=modules,
                                max_cascades=max_cascades,
                                )
    prepopulator.prepopulate()


if __name__ == '__main__':
    fire.Fire(run_prepopulator)
    # run_prepopulator(modules=['diffusion'], config_file='dev_config.yaml', max_cascades=23)
