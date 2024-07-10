from abc import ABC


class BasePropagationMetrics(ABC):
    def __init__(self, host='localhost', port=27017, reference_types=('retweeted', 'quoted')):
        self.reference_types = reference_types
        self.host = host
        self.port = port

    def persist(self, datasets):
        pass

    def load_from_mongodb(self, datasets):
        pass
