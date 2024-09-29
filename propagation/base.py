from abc import ABC


class BasePropagationMetrics(ABC):
    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted', 'replied_to')):
        self.reference_types = reference_types
        self.host = host
        self.port = port

    def persist(self, datasets):
        pass

    def _validate_dataset(self, client, dataset):
        if dataset not in client.list_database_names():
            raise RuntimeError(f'Dataset {dataset} not found')
        else:
            collections = client.get_database(dataset).list_collection_names()
            if 'raw' not in collections:
                raise RuntimeError(f'Collection raw not found in dataset {dataset}')
