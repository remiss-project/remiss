import shutil
import unittest
from pathlib import Path

from pymongo import MongoClient

from preprocess import preprocess_multimodal_dataset_data
from tests.conftest import populate_test_database, delete_test_database


class MyTestCase(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     populate_test_database('test_dataset')
    #
    # @classmethod
    # def tearDownClass(cls):
    #     delete_test_database('test_dataset')
    #     shutil.rmtree('/tmp/test_cache')

    def test_prepopulate(self):
        config = {'mongodb': {'host': 'localhost', 'port': 27017},
                     'cache_dir': '/tmp/test_cache',
                     'graph_layout': 'fruchterman_reingold',
                     'graph_simplification': {'method': 'backbone', 'threshold': 0.5},
                     'frequency': '1D',
                     'available_datasets': None,
                     'prepopulate': False}
        with open('/tmp/test_config.yaml', 'w') as f:
            f.write(str(config).replace('None', ''))
        prepopulate('/tmp/test_config.yaml')
        # Check that data has been actually populated
        client = MongoClient('localhost', 27017)
        db = client['test_dataset']
        self.assertGreater(db['user_propagation'].count_documents({}), 0)
        self.assertGreater(db['conversation_propagation'].count_documents({}), 0)

        # Check that the cache has been populated
        self.assertTrue(list(Path('/tmp/test_cache/test_dataset').glob('*.graphmlz'))[0].exists())

    def test_preprocess_multimodal(self):
        preprocess_multimodal_dataset_data('../remiss_data_share', 'multimodal_data')


if __name__ == '__main__':
    unittest.main()
