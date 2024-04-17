from pathlib import Path

import pymongoarrow
from pymongo import MongoClient

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()


class FactCheckingPlotFactory(MongoPlotFactory):
    def __init__(self, data_dir='fact_checking_data', host="localhost", port=27017, database="test_remiss",
                 fact_checking_database='fact_checking',
                 available_datasets=None):
        super().__init__(host, port, database, available_datasets)
        self.fact_checking_database = fact_checking_database
        self.data_dir = Path(data_dir)

    def plot_fact_checking(self, dataset, tweet_id):
        data_id = self.get_fact_checking_data_id(dataset, tweet_id)
        data = self.load_data_for_tweet(dataset, data_id)
        return data

    def get_fact_checking_data_id(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.fact_checking_database)
        self._validate_collection(database, dataset)
        collection = database.get_collection(dataset)
        data_id = collection.find_one({'tweet_id': tweet_id})['id']
        client.close()
        return data_id

    def load_data_for_tweet(self, dataset, data_id):
        data_path = self.data_dir / dataset / f'{data_id}.htm'
        with open(data_path, 'r') as f:
            data = f.read()

        return data







# Features
# - Claim
#    - Tweet text (T)
#    - Tweet images (V)
# - Evidence
#    - Visual evidences (images obtained from text of the tweet) (XV)
#    - Textual evidences (text obtained from the images of tweet) (XT)
# - Two measurements
#   - claim text to evidence text (T vs XT)
#     Build a graph and check graph structure similarity (graph match only text).
#     The stuff going in the graph are already preprocessed in order to filter irrelevant stuff with
#     conditional filtering using LLM's
#
#   - claim image to evidence image (V vs XV)
#     Five metrics + cosine similarity with at least a 3 out of 5 with greater than 0.9
#      - Semantic
#      - Place
#      - Face
#      - Object
#      - Automatic Caption
# Things to display
# - T vs VT graph comparison
#   - T graph
#   - XT graph
#   - XV(T) context graph
#   - metrics
#     - number of supported egdes (whatever claims appear they are also in the evidence). 30% must be supported.
#     - number of conflicted nodes (the nodes have any conflict). there must be no conflicts.
# - visual five metrics for V vs VX
# - three images
# INPUTS
#   - V  -> [XT] -> XT
#   - T -> [XV] -> XV
#  OUTPUTS:
#     image {claim_graph_annotated
#     xt_graph
#     xv_graph}
#     numbers {5 visual scores, 2 graph scores}
