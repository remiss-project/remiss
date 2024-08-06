import logging

from pymongo import MongoClient

from propagation import DiffusionMetrics, NetworkMetrics, Egonet

logger = logging.getLogger(__name__)


class PropagationPreprocessor:
    def __init__(self, dataset, data, host='localhost', port=27017, reference_types=('retweeted', 'quoted')):
        self.dataset = dataset
        self.data = data
        self.host = host
        self.port = port
        self.reference_types = reference_types

        self.diffusion_metrics = DiffusionMetrics(host=self.host, port=self.port, reference_types=self.reference_types)
        self.network_metrics = NetworkMetrics(host=self.host, port=self.port, reference_types=self.reference_types)
        self.egonet = Egonet(host=self.host, port=self.port, reference_types=self.reference_types)

    def process(self):
        logger.info(f"Processing dataset {self.dataset} with data {self.data}")
        # Send data to raw collection
        logger.info(f"Storing data in raw collection")
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        database.drop_collection('raw')
        collection = database.get_collection('raw')
        collection.insert_many(self.data)
        logger.info('Generating diffusion metrics')
        self.diffusion_metrics.persist([self.dataset])
        logger.info('Generating network metrics')
        self.network_metrics.persist([self.dataset])
        logger.info('Generating egonet metrics')
        self.egonet.persist([self.dataset])
