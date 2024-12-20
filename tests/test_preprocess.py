import json
import random
from collections.abc import Mapping
from unittest import TestCase

import numpy as np
import pandas as pd
from twarc import ensure_flattened

from preprocess import preprocess_tweets, fix_timestamps


def extract_paths(base_path, dd):
    new_paths = []
    for key, value in dd.items():
        new_path = base_path + ('.' if base_path else '') + key
        if isinstance(value, Mapping):
            new_paths.extend(extract_paths(new_path, value))
        else:
            new_paths.append(new_path)
    return new_paths


def generate_test_data(start_date, end_date, hashtags, parties,
                       num_tweets, prob_hashtag, prob_party, prob_usual_suspects, prob_interaction,
                       prob_rt, prob_qt, prob_reply,
                       output_file):
    dates = pd.date_range(start_date, end_date, periods=num_tweets)
    with open(output_file, 'w') as outfile:
        for i in range(num_tweets):
            tweet = {'id': i,
                     'created_at': {'$date': dates[i].replace(microsecond=0).isoformat() + 'Z'},
                     'author': {'username': f'TEST_USER_{i // 2}',
                                'id': i // 2,
                                'remiss_metadata': {}
                                }}
            if np.random.rand() < prob_hashtag:
                tweet['entities'] = {'hashtags': [{'tag': np.random.choice(hashtags)}]}
            else:
                tweet['entities'] = {'hashtags': []}
            if np.random.rand() < prob_party:
                tweet['author']['remiss_metadata']['party'] = np.random.choice(parties)
            else:
                tweet['author']['remiss_metadata']['party'] = None

            if np.random.rand() < prob_usual_suspects:
                tweet['author']['remiss_metadata']['is_usual_suspect'] = True
            else:
                tweet['author']['remiss_metadata']['is_usual_suspect'] = False

            if np.random.rand() < prob_interaction:
                interaction_type = random.choices(['replied_to', 'retweeted', 'quoted'],
                                                  [prob_reply, prob_rt, prob_qt], k=1)[0]
                tweet['referenced_tweets'] = [{'id': np.random.randint(0, num_tweets),
                                               'type': interaction_type}]

            outfile.write(json.dumps(tweet) + '\n')


class TestPreprocess(TestCase):
    """
    Some tests assume the existence of a couple of test collections imported like this from the data generated by
    preprocess_tweets.
    mongoimport --db test_remiss --collection test_tweets --file import_test.mongodbimport.jsonl
    mongoimport --db test_remiss --collection test_tweets_2 --file import_test.mongodbimport.jsonl
    """

    def _test_generate_test_data(self):
        hashtags = ['EspañaVaciada', 'EspañaViva', '28A', 'PedroSánchez', 'Podemos', 'Ferraz', 'LaHistoriaLaEscribesTú',
                    'PabloIglesias', 'CataluñaPorEspaña', 'cloacas', 'objetivoiglesias', 'LlibertatPresosPolitics',
                    'StopRepresion', 'ObjetivoIglesias', 'EspanaViva', 'EspanaVaciada', 'LasPalmas', 'Vox', 'RT',
                    'barcelona', 'VOX', 'LaEspañaViva', 'Cuenca', 'PP', 'ValorSeguro', 'Ciudadanos',
                    'LaEspañaQueQuieres', 'elecciones', 'UnidasPodemos', 'AhoraSiVamosBien', 'Madrid', 'VOXSaleAGanar',
                    'despoblación', 'JoAcuso', '26M', 'YoVotoUnidasPodemos', 'SiguemeYTeSigoVOX', 'EspañaVACIADA',
                    'Españaviva', 'ESPAÑAVACIADA', 'LaSilenciosaCat', 'LlibertatPresosPolítics', 'VÍDEO', 'AFD',
                    '100MedidasVOX', 'España', 'clocas', '28Abril', 'EspañaValiente', 'VolerelaLuna', '1O', 'PSOE',
                    'Barcelona', 'CatalunaPorEspaña', 'Jaén', '20S', 'ContraLaDespoblación', 'MedioRural',
                    'nacionalpopulismo', 'PorEspaña', 'MentirasMediosVerdadesVOX', 'EspañolesPrimero', 'Casado',
                    'SonTerroristas', 'JaenLevantateBrava', '31M', 'L6NAlbertRivera', 'VOXEnTodaEspaña',
                    'JudiciDemocràcia', 'FelizDomingo', 'Españavaciada', 'Europa', 'Sanchismo', 'EspañaDespierta',
                    'Bannon', 'podemos', 'Llibertatpresospolitics', 'españavaciada', 'Huesca', 'JaénMereceMás',
                    'CustodiaCompartida', 'Policía', 'espiada', 'Trifachito', 'VoxEnTodaEspaña', 'presupuestos',
                    'Cataluña', 'vota', 'Revuelta', '31MsíVoy', '8M', 'VoxAvanzaEnValencia', 'VallsBCN2019', 'Aragón',
                    'VandanaShiva', 'BonaNit', 'judiciprocés', 'GolpistasAPrisión', 'Absolució', 'NoPasarán']
        parties = ['PSOE', 'PP', 'Cs', 'UP', 'VOX', 'ERC', 'JxCat', 'PNV', 'Bildu', 'CUP', 'CC', 'NA+', 'PRC', 'BNG']
        start_date = pd.to_datetime('2019-01-01 23:20:00')
        end_date = pd.to_datetime('2020-12-31 23:59:59')
        num_tweets = 10000
        prob_hashtag = 0.5
        prob_party = 0.2
        prob_usual_suspects = 0.2
        prob_interaction = 0.8
        prob_rt = 0.9
        prob_qt = 0.05
        prob_reply = 0.05
        output_file = 'test_resources/test.jsonl'
        generate_test_data(start_date, end_date, hashtags, parties, num_tweets, prob_hashtag, prob_party,
                           prob_interaction,
                           prob_rt, prob_qt, prob_reply,
                           prob_usual_suspects, output_file)

    def test_preprocess_tweets_2(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip',
                          metadata_file='test_resources/test_metadata.xlsx')
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            processed = [json.loads(line) for line in f]
        with open('test_resources/test_original.jsonl') as f:
            original = [json.loads(line) for line in f]

        original = [ensure_flattened(t) for t in original]
        original = sum([ensure_flattened(t) for t in original], [])
        for p, o in zip(processed, original):
            self.assertEqual(p['id'], o['id'])
            self.assertEqual(p['created_at'], o['created_at'])
            self.assertEqual(p['author']['username'], o['author']['username'])
            self.assertEqual(p['author']['id'], o['author']['id'])
            self.assertEqual(p['author']['name'], o['author']['name'])
            self.assertEqual(p['author']['description'], o['author']['description'])
            self.assertEqual(p['text'], o['text'])

    def test_preprocess_tweets_with_media(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip',
                          metadata_file='test_resources/test_metadata.xlsx')
        # check that every tweet with media has a corresponding entry in the media file
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            tweets = [json.loads(line) for line in f]
        with open('test_resources/test_original.media.jsonl') as f:
            media = [json.loads(line) for line in f]

        expected_text = [t['text'] for t in tweets if 'attachments' in t and 'media' in t['attachments']]
        actual_text = [m['text'] for m in media]
        self.assertEqual(expected_text, actual_text)

        expected_media = [t['attachments']['media'] for t in tweets
                          if 'attachments' in t and 'media' in t['attachments']]
        actual_media = [m['media'] for m in media]
        self.assertEqual(expected_media, actual_media)

        expected_author_data = [t['author'] for t in tweets if 'attachments' in t and 'media' in t['attachments']]
        actual_author_data = [m['author'] for m in media]
        self.assertEqual(expected_author_data, actual_author_data)

        expected_remiss_metadata = [t['author']['remiss_metadata'] for t in tweets
                                    if 'attachments' in t and 'media' in t['attachments']]

        actual_remiss_metadata = [m['author']['remiss_metadata'] for m in media]
        self.assertEqual(expected_remiss_metadata, actual_remiss_metadata)

        expected_id = [t['id'] for t in tweets if 'attachments' in t and 'media' in t['attachments']]
        actual_id = [m['id'] for m in media]
        self.assertEqual(expected_id, actual_id)

    def test_preprocess_tweets_with_metadata(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip', metadata_file='test_resources/test_metadata.xlsx')
        # check that every tweet with media has a corresponding entry in the media file
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            tweets = [json.loads(line) for line in f]

        expected_usual_suspects = pd.read_excel('test_resources/test_metadata.xlsx',
                                                sheet_name='NOVA LLISTA USUAL SUSPECTS')
        expected_usual_suspects = set(
            expected_usual_suspects['ENLLAÇ'].str.split('/').str[-1].str.split('?').str[0].to_list())
        expected_politicians = pd.read_excel('test_resources/test_metadata.xlsx', sheet_name='LLISTA POLÍTICS')
        expected_politicians = set(
            expected_politicians['ENLLAÇ TW'].str.split('/').str[-1].str.split('?').str[0].to_list())
        actual_usual_suspects = set(
            [t['author']['username'] for t in tweets if t['author']['username'] in expected_usual_suspects])
        actual_politicians = set(
            [t['author']['username'] for t in tweets if t['author']['username'] in expected_politicians])
        self.assertEqual(expected_usual_suspects, actual_usual_suspects)
        self.assertEqual(expected_politicians, actual_politicians)

    def test_preprocess_usual_suspects_and_politicians(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip', metadata_file='test_resources/test_metadata.xlsx')
        # check that every tweet from an usual suspect or politician is present at usual_suspects_and_politicians.jsonl
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            tweets = [json.loads(line) for line in f]
        with open('test_resources/test_original.usual_suspects_and_politicians.jsonl') as f:
            sus = [json.loads(line) for line in f]

        expected_ids = [t['id'] for t in sus]
        sus_ids = [t['id'] for t in tweets if t['author']['remiss_metadata']['is_usual_suspect']
                   or t['author']['remiss_metadata']['party']]
        self.assertEqual(expected_ids, sus_ids)

    def test_preprocess_timestamps(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip',
                          metadata_file='test_resources/test_metadata.xlsx')
        # retrieve all the fields that are timestamps
        date_fields = ['created_at', 'editable_until', 'retrieved_at']

        def assert_mongoimport_date_format(tweet):
            date_fields = {'created_at', 'editable_until', 'retrieved_at'}
            for field, value in tweet.items():
                if field in date_fields:
                    self.assertIsInstance(value, dict)
                    self.assertEqual(len(value), 1)
                    self.assertEqual(list(value.keys()), ['$date'])
                    date = list(value.values())
                    self.assertEqual(len(date), 1)
                    date_str = date[0]
                    # check that the date_str is an actual iso8601 date
                    pd.to_datetime(date_str)

                elif isinstance(value, dict):
                    assert_mongoimport_date_format(value)

        with open('test_resources/test_original.mongoimport.jsonl') as f:
            for line in f:
                tweet = json.loads(line)
                # find all nested fields that contain timestamps
                assert_mongoimport_date_format(tweet)

    def test_preprocess_timestamps_2(self):
        def assert_mongoimport_date_format(tweet):
            date_fields = {'created_at', 'editable_until', 'retrieved_at'}
            for field, value in tweet.items():
                if field in date_fields:
                    self.assertIsInstance(value, dict, f'Field {field} is not a dict')
                    self.assertEqual(len(value), 1)
                    self.assertEqual(list(value.keys()), ['$date'])
                    date = list(value.values())
                    self.assertEqual(len(date), 1)
                    date_str = date[0]
                    # check that the date_str is an actual iso8601 date
                    pd.to_datetime(date_str)

                elif isinstance(value, dict):
                    assert_mongoimport_date_format(value)
                elif isinstance(value, list):
                    for v in value:
                        if isinstance(v, dict):
                            assert_mongoimport_date_format(v)

        with open('test_resources/test_preprocess_dates.json') as f:
            tweet = json.load(f)
            fix_timestamps(tweet)
            assert_mongoimport_date_format(tweet)
