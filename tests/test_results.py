import datetime
import random
import shutil
import unittest
import uuid
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import Timestamp
from pymongo import MongoClient
from tqdm import tqdm

from propagation.results import Results


class ResultsTestCase(unittest.TestCase):
    def setUp(self):
        self.results = Results(datasets=['test_dataset_2'], host='localhost', port=27017,
                               output_dir=Path('../results'), top_n=1)
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = str(uuid.uuid4().hex)
        self.test_tweet_id = '1167074391315890176'
        self.missing_tweet_id = '1077146799692021761'

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def test_plot_propagation_tree(self):
        self.results.plot_propagation_tree()
        # check the plot is there
        files = list(self.results.output_dir.glob('propagation_tree*.png'))
        self.assertTrue(files)

    def test_plot_egonet_and_backbone(self):
        self.results.plot_egonet_and_backbone()
        # check the plot is there
        files = list(self.results.output_dir.glob('egonet*.png'))
        self.assertTrue(files)

    def test_plot_legitimacy_reputation_and_status(self):
        self.results.plot_legitimacy_status_and_reputation()
        # check the plot is there
        files = list(self.results.output_dir.glob('legitimacy*.png'))
        self.assertTrue(files)
        files = list(self.results.output_dir.glob('reputation*.png'))
        self.assertTrue(files)
        files = list(self.results.output_dir.glob('status*.png'))
        self.assertTrue(files)

    def test_cascades(self):
        self.results.plot_cascades()
        # check the plot is there
        files = list(self.results.output_dir.glob('normal_cascade*.png'))
        self.assertTrue(files)
        # files = list(self.results.output_dir.glob('suspect_cascade*.png'))
        # self.assertTrue(files)
        files = list(self.results.output_dir.glob('politician_cascade*.png'))
        self.assertTrue(files)
        # files = list(self.results.output_dir.glob('suspect_politician_cascade*.png'))
        # self.assertTrue(files)

    def test_nodes_edges(self):
        self.results.generate_nodes_and_edges_table()
        # check the plot is there
        files = list(self.results.output_dir.glob('nodes_edges.csv'))
        self.assertTrue(files)

if __name__ == '__main__':
    unittest.main()
