import shutil
import unittest
from pathlib import Path

from pymongo import MongoClient

from preprocess import preprocess_multimodal_dataset_data
from tests.conftest import populate_test_database, delete_test_database


class MyTestCase(unittest.TestCase):
    def test_preprocess_multimodal(self):
        preprocess_multimodal_dataset_data('../remiss_data_share', 'multimodal_data')


if __name__ == '__main__':
    unittest.main()
