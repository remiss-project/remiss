import unittest

from plotly.graph_objs import Figure
from pymongo import MongoClient

from figures.fact_checking import FactCheckingPlotFactory

DATA_DIR = './../fact_checking_data'


class TestTimeSeriesFactory(unittest.TestCase):
    def setUp(self):
        self.plot_factory = FactCheckingPlotFactory(data_dir=DATA_DIR)
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_dataset')
        self.database = self.client.get_database('test_dataset')
        self.collection = self.database.get_collection('multimodal')
        self.test_data = [
            {
                "claim_text": "A shot of the All Nippon Airways Boeing 787 Dreamliner that s painted in the likeness of R2D2 in Los Angeles on Dec 15 2015",
                "id": 47,
                "tweet_id": "100485425",
                "text_evidences": "-\n ANA's R2D2 Jet Uses The Force to Transport Stars Between The 'Star \nWars' Premieres - TheDesignAir\n\n- The Cast Of \"Star Wars: The Force Awakens\" On ANA Charter Flight From \nLos Angeles To The London Premiere\n\n- The R2-D2 ANA Jet Transports Star Wars Movie Cast Between Premieres in\n USA and UK\n\n- Dec15.32\n\n- 24 Boeing 787 ideas | boeing 787, boeing, boeing 787 ... - Pinterest\n\n- The stars of \"Star Wars: The Force Awakens\" blew into London in It \nMovie Cast, It Cast, Geek Movies, Star Wars Cast, Private Pilot, Air \nPhoto, Airplane Design, Aircraft Painting, Commercial Aircraft\n\n- 19 Geek Stuff ideas | geek stuff, star wars, stars\n\n- 100 Aviation ideas | aviation, boeing, aircraft\n",
                "evidence_text": "The Cast Of \"Star Wars: The Force Awakens\" On ANA Charter Flight From Los Angeles To The London Premiere",
                "evidence_image_alt_text": "Page\n 2 - R2d2 Star Wars High Resolution Stock Photography and Images - Alamy\n Page 2 - R2d2 Star Wars High Resolution Stock Photography and ...",
                "results": {
                    "predicted_label": 1,
                    "actual_label": 0,
                    "num_claim_edges": 5,
                    "frac_verified": 0.0,
                    "explanations": "+ XT(V) ns + XV(T) ns",
                    "visual_similarity_score": 0.8824891924858094
                }
            }
        ]
        self.collection.insert_many(self.test_data)
        self.client.close()

    def tearDown(self) -> None:
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_dataset')
        self.client.close()

    def tearDown(self) -> None:
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database('test_dataset')
        self.client.close()

    def test_claim_image(self):
        plot = self.plot_factory.plot_claim_image('test_dataset', '100485425')
        plot.show()
        assert isinstance(plot, Figure)

    def test_evidence_image(self):
        plot = self.plot_factory.plot_evidence_image('test_dataset', '100485425')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_claim(self):
        plot = self.plot_factory.plot_graph_claim('test_dataset', '100485425')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_evidence_text(self):
        plot = self.plot_factory.plot_graph_evidence_text('test_dataset', '100485425')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_evidence_visual(self):
        plot = self.plot_factory.plot_graph_evidence_vis('test_dataset', '100485425')
        plot.show()
        assert isinstance(plot, Figure)

    def test_visual_evidences(self):
        plot = self.plot_factory.plot_visual_evidences('test_dataset', '100485425')
        plot.show()
        assert isinstance(plot, Figure)

    def test_load_data_for_tweet(self):
        data = self.plot_factory.load_data_for_tweet('test_dataset', '100485425')
        assert data == self.test_data[0]