import time

from app import load_config
from figures.propagation import PropagationPlotFactory


def prepopulate(config_file='dev_config.yaml'):
    config = load_config(config_file)
    propagation_plot_factory = PropagationPlotFactory(host=config['mongodb']['host'], port=config['mongodb']['port'],
                                                      layout=config['graph_layout'],
                                                      threshold=config['graph_simplification']['threshold'],
                                                      frequency=config['frequency'],
                                                      available_datasets=config['available_datasets'])

    print('Prepopulating...')
    start_time = time.time()
    propagation_plot_factory.prepopulate()

    print(f'Prepopulated in {time.time() - start_time} seconds.')

if __name__ == '__main__':
    prepopulate()