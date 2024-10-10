import shutil
import unittest
import uuid
from pathlib import Path

from plotly.graph_objs import Figure
from pymongo import MongoClient

from figures.multimodal import MultimodalPlotFactory

# @unittest.skip("Skip test")
class TestMultimodalPlotFactory(unittest.TestCase):

    def tearDown(self):
        client = MongoClient('localhost', 27017)
        client.drop_database(self.tmp_dataset)

    def setUp(self):
        self.test_dataset = 'test_dataset_2'
        self.tmp_dataset = 'tmp_dataset'
        self.client = MongoClient('localhost', 27017)
        self.client.drop_database(self.tmp_dataset)
        self.database = self.client.get_database(self.tmp_dataset)
        self.collection = self.database.get_collection('multimodal')
        self.test_data = [
            {
                "visual_evidence_domain": "elpais.com",
                "visual_evidence_matched_categories": "['caption', 'place', 'vit', 'objects']",
                "visual_evidence_text": " El candidato a lehendakari del PNV, Imanol Pradales, comparece en la sede del PNV tras el conteo, este domingo.Resultados de las elecciones vascas 2024 | El PNV empata con Bildu y podr\u00e1 reeditar la coalici\u00f3n con los socialistas | Elecciones en el Pa\u00eds Vasco 21-A | EL PA\u00cdS",
                "visual_evidence_similarity_score": "0.533111073076725",
                "visual_evidence_graph_similarity_score": "1.0",
                "text_evidence": "no_text_evidence",
                "text_evidence_similarity_score": "0.4457797110080719",
                "text_evidence_graph_similarity_score": "0.0",
                "visual_evidence_domain1": "elpais.com",
                "visual_evidence_matched_categories1": "['caption', 'place', 'vit', 'objects']",
                "visual_evidence_text1": " El candidato a lehendakari del PNV, Imanol Pradales, comparece en la sede del PNV tras el conteo, este domingo.Resultados de las elecciones vascas 2024 | El PNV empata con Bildu y podr\u00e1 reeditar la coalici\u00f3n con los socialistas | Elecciones en el Pa\u00eds Vasco 21-A | EL PA\u00cdS",
                "visual_evidence_similarity_score1": "0.5365476682782173",
                "visual_evidence_graph_similarity_score1": "1.0",
                "claim_text": "Desde hace tiempo se ve\u00eda venir que se pod\u00eda vertebrar una coalici\u00f3n como en Euskadi @eajpnv + @socialistavasco en Nafarroa @PSNPSOE + @geroabai \nCon estas declaraciones queda claro que esa es la estrategia nacional del PNV en toda Euskal Herria.\n#26M https://t.co/nAOIdpq1YB",
                "found_flag": "not found",
                "id_in_json": 7620,
                "t_sug": "Estrategia Nacional PNV Euskal Herria EAJPNV SocialistaVasco Navarra",
                "old_t_sug": "coalici\u00f3n Euskadi Euskal Herria estrategia nacional PNV",
                "results": {
                    "predicted_label": "FAKE",
                    "actual_label": "FAKE",
                    "visual_similarity_score": 0.533111073076725,
                    "explanations": ""
                },
                "tweet_id": "1133352119124353024"
            },
        ]
        self.collection.insert_many(self.test_data)
        self.client.close()

        test_data_dir = Path('./test_resources/multimodal')
        data_dir = Path('/tmp/multimodal_data') / self.tmp_dataset
        shutil.copytree(test_data_dir, data_dir)
        self.plot_factory = MultimodalPlotFactory(data_dir=data_dir.parent, available_datasets=[self.tmp_dataset])

    def test_claim_image(self):
        plot = self.plot_factory.plot_claim_image(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_claim(self):
        plot = self.plot_factory.plot_graph_claim(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_evidence_vis(self):
        plot = self.plot_factory.plot_graph_evidence_vis(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_evidence_text(self):
        plot = self.plot_factory.plot_graph_evidence_text(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_evidence_image(self):
        plot = self.plot_factory.plot_evidence_image(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_claim1(self):
        plot = self.plot_factory.plot_graph_claim1(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_graph_evidence_vis1(self):
        plot = self.plot_factory.plot_graph_evidence_vis1(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_plot_graph_evidence_text1(self):
        plot = self.plot_factory.plot_graph_evidence_text1(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_plot_evidence_image1(self):
        plot = self.plot_factory.plot_evidence_image1(self.tmp_dataset, '1133352119124353024')
        plot.show()
        assert isinstance(plot, Figure)

    def test_load_data_for_tweet(self):
        data = self.plot_factory.load_data_for_tweet(self.tmp_dataset, '1133352119124353024')
        assert data == self.test_data[0]
