from pathlib import Path

import plotly.express as px
import pymongoarrow
from pymongo import MongoClient
from skimage import io

from figures.figures import MongoPlotFactory

pymongoarrow.monkey.patch_all()


class MultimodalPlotFactory(MongoPlotFactory):
    def __init__(self, data_dir='multimodal_data', host="localhost", port=27017,
                 multimodal_collection_name='multimodal',
                 available_datasets=None):
        super().__init__(host, port, available_datasets)
        self.multimodal_collection_name = multimodal_collection_name
        self.data_dir = Path(data_dir)

    def plot_claim_image(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        fig = self.load_image(dataset, data['id'], 'claim_image')
        return fig

    def plot_evidence_image(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        fig = self.load_image(dataset, data['id'], 'evidence_image')
        return fig

    def plot_graph_claim(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        fig = self.load_image(dataset, data['id'], 'graph_claim')
        return fig

    def plot_graph_evidence_text(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        fig = self.load_image(dataset, data['id'], 'graph_evidence_text')
        return fig

    def plot_graph_evidence_vis(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        fig = self.load_image(dataset, data['id'], 'graph_evidence_vis')
        return fig

    def plot_visual_evidences(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        fig = self.load_image(dataset, data['id'], 'visual_evidences')
        return fig

    def get_metadata(self, dataset, tweet_id):
        data = self.load_data_for_tweet(dataset, tweet_id)
        return data

    def load_image(self, dataset, fact_checking_id, image_type):
        image_dir = self.data_dir /  dataset / 'images' / str(fact_checking_id)
        # find matching image with image_type as filename, disregarding extension
        try:
            image_path = next(image_dir.glob(f'{image_type}.*'))
        except StopIteration:
            raise RuntimeError(f'Image {image_type} not found for fact checking id {fact_checking_id}')
        img = io.imread(image_path)
        fig = px.imshow(img)
        return fig

    def load_data_for_tweet(self, dataset, tweet_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(dataset)
        self._validate_dataset(client, dataset)
        collection = database.get_collection(self.multimodal_collection_name)
        data = collection.find_one({'tweet_id': tweet_id})
        client.close()
        if data is None:
            raise RuntimeError(f'Tweet {tweet_id} not found in dataset {dataset}')
        return data

    def _validate_dataset(self, client, dataset):
        if dataset not in client.list_database_names():
            raise RuntimeError(f'Dataset {dataset} not found')
        else:
            collections = client.get_database(dataset).list_collection_names()
            if self.multimodal_collection_name not in collections:
                raise RuntimeError(f'Collection {self.multimodal_collection_name} not found in dataset {dataset}')
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