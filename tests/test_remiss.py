import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from unittest import TestCase

from pymongo import MongoClient

from app import update_graph
from remiss import preprocess_tweets, load_tweet_count_evolution
import pandas as pd
import plotly.express as px
import numpy as np


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
                       num_tweets, prob_hashtag, prob_party, prob_usual_suspects,
                       output_file):
    dates = pd.date_range(start_date, end_date, periods=num_tweets)
    with open(output_file, 'w') as outfile:
        for i in range(num_tweets):
            tweet = {'id': i,
                     'created_at': {'$date': dates[i].replace(microsecond=0).isoformat() + 'Z'},
                     'author': {'username': f'TEST_USER_{i}',
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

            outfile.write(json.dumps(tweet) + '\n')


class TestRemiss(TestCase):
    """
    Some tests assume the existance of a couple of test collections imported like this from the data generated by
    preprocess_tweets.
    mongoimport --db test_remiss --collection test_tweets --file import_test.mongodbimport.jsonl
    mongoimport --db test_remiss --collection test_tweets_2 --file import_test.mongodbimport.jsonl
    """

    def test_preprocess_tweets(self):
        return
        preprocess_tweets('test_resources/test_original.jsonl.zip')
        # check number of lines is correct
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            self.assertEqual(len(f.readlines()), 934)

        expected_keys = ['id', 'conversation_id', 'referenced_tweets.replied_to.id', 'referenced_tweets.retweeted.id',
                         'referenced_tweets.quoted.id', 'author_id', 'in_reply_to_user_id', 'in_reply_to_username',
                         'retweeted_user_id', 'retweeted_username', 'quoted_user_id', 'quoted_username', 'created_at',
                         'text', 'lang', 'source', 'public_metrics.impression_count', 'public_metrics.reply_count',
                         'public_metrics.retweet_count', 'public_metrics.quote_count', 'public_metrics.like_count',
                         'public_metrics.bookmark_count', 'reply_settings', 'edit_history_tweet_ids',
                         'edit_controls.edits_remaining', 'edit_controls.editable_until',
                         'edit_controls.is_edit_eligible', 'possibly_sensitive', 'withheld.scope',
                         'withheld.copyright', 'withheld.country_codes', 'entities.annotations', 'entities.cashtags',
                         'entities.hashtags', 'entities.mentions', 'entities.urls', 'context_annotations',
                         'attachments.media', 'attachments.media_keys', 'attachments.poll.duration_minutes',
                         'attachments.poll.end_datetime', 'attachments.poll.id', 'attachments.poll.options',
                         'attachments.poll.voting_status', 'attachments.poll_ids', 'author.id', 'author.created_at',
                         'author.username', 'author.name', 'author.description',
                         'author.entities.description.cashtags', 'author.entities.description.hashtags',
                         'author.entities.description.mentions', 'author.entities.description.urls',
                         'author.entities.url.urls', 'author.url', 'author.location', 'author.pinned_tweet_id',
                         'author.profile_image_url', 'author.protected', 'author.public_metrics.followers_count',
                         'author.public_metrics.following_count', 'author.public_metrics.listed_count',
                         'author.public_metrics.tweet_count', 'author.verified', 'author.verified_type',
                         'author.withheld.scope', 'author.withheld.copyright', 'author.withheld.country_codes',
                         'geo.coordinates.coordinates', 'geo.coordinates.type', 'geo.country', 'geo.country_code',
                         'geo.full_name', 'geo.geo.bbox', 'geo.geo.type', 'geo.id', 'geo.name', 'geo.place_id',
                         'geo.place_type', 'matching_rules', '__twarc.retrieved_at', '__twarc.url', '__twarc.version']

        # check that the file is valid jsonl and the keys are correct
        expected_keys = set(expected_keys)
        actual_keys = set()
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            for line in f:
                tweet = json.loads(line)
                actual_keys.update(extract_paths('', tweet))

        # SKIPPING THIS TEST AS THE CONVERTER USED TO GET THE KEYS ADDS SOME MODIFICATIONS OF ITS OWN AND WE WANT
        # TO PRESERVE THE ORIGINAL STRUCTURE
        # self.assertEqual(expected_keys, actual_keys)
        expected_missing = {'in_reply_to_user.entities.description.hashtags', 'in_reply_to_user.location',
                            'withheld.country_codes', 'withheld.copyright',
                            'in_reply_to_user.entities.description.mentions', 'in_reply_to_user.name',
                            'in_reply_to_user.username', 'referenced_tweets.retweeted.id', 'in_reply_to_user.verified',
                            'public_metrics.bookmark_count', 'in_reply_to_user.description', 'source',
                            'author.entities.description.cashtags', 'in_reply_to_user.verified_type',
                            'in_reply_to_user.entities.url.urls', 'in_reply_to_user.url', 'geo.coordinates.coordinates',
                            'referenced_tweets.quoted.id', 'withheld.scope', 'quoted_username',
                            'author.withheld.country_codes', 'entities.cashtags', 'author.withheld.copyright',
                            'in_reply_to_user.protected', 'matching_rules', 'author.withheld.scope',
                            'in_reply_to_user.public_metrics.tweet_count', 'in_reply_to_username',
                            'in_reply_to_user.public_metrics.following_count', 'geo.coordinates.type',
                            'in_reply_to_user.profile_image_url', 'in_reply_to_user.created_at',
                            'in_reply_to_user.pinned_tweet_id', 'in_reply_to_user.public_metrics.followers_count',
                            'in_reply_to_user.public_metrics.listed_count', 'retweeted_username', 'quoted_user_id',
                            'in_reply_to_user.entities.description.urls', 'referenced_tweets', 'in_reply_to_user.id',
                            'referenced_tweets.replied_to.id', 'retweeted_user_id'}

        actual_missing = expected_keys - actual_keys | actual_keys - expected_keys
        self.assertEqual(expected_missing, actual_missing)
        Path('test_resources/test_original.preprocessed.jsonl').unlink()
        Path('test_resources/test_original.media.jsonl').unlink()
        Path('test_resources/test_original.mongodbimport.jsonl').unlink()

    def test_preprocess_tweets_with_media(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip')
        # check that every tweet with media has a corresponding entry in the media file
        with open('test_resources/test_original.preprocessed.jsonl') as f:
            tweets = [json.loads(line) for line in f]
        with open('test_resources/test_original.media.jsonl') as f:
            media = [json.loads(line) for line in f]

        media_ids = set([m['id'] for m in media])
        tweets_with_media = [t for t in tweets if 'attachments' in t and 'media' in t['attachments']]
        tweets_with_media_ids = set([t['id'] for t in tweets_with_media])
        self.assertEqual(media_ids, tweets_with_media_ids)
        Path('test_resources/test_original.preprocessed.jsonl').unlink()
        Path('test_resources/test_original.media.jsonl').unlink()
        Path('test_resources/test_original.mongodbimport.jsonl').unlink()

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
        Path('test_resources/test_original.preprocessed.jsonl').unlink()
        Path('test_resources/test_original.media.jsonl').unlink()
        Path('test_resources/test_original.mongodbimport.jsonl').unlink()

    def test_preprocess_timestamps(self):
        preprocess_tweets('test_resources/test_original.jsonl.zip')
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

        with open('test_resources/test_original.mongodbimport.jsonl') as f:
            for line in f:
                tweet = json.loads(line)
                # find all nested fields that contain timestamps
                assert_mongoimport_date_format(tweet)

        Path('test_resources/test_original.mongodbimport.jsonl').unlink()
        Path('test_resources/test_original.preprocessed.jsonl').unlink()
        Path('test_resources/test_original.media.jsonl').unlink()

    def test_load_tweet_count_evolution(self):

        client = MongoClient("localhost", 27017)
        database = client.get_database("test_remiss")
        collection = database.get_collection('test_tweets')
        df = pd.DataFrame(list(collection.find()))
        expected = df.groupby(pd.Grouper(key='created_at', freq='1D')).size()
        actual = load_tweet_count_evolution('localhost', 27017, 'test_remiss', 'test_tweets',
                                            unit='day', bin_size=1)
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_load_tweet_count_evolution_start_end_date(self):
        start_date = pd.to_datetime('2019-01-01 23:20:00')
        end_date = pd.to_datetime('2020-12-31 23:59:59')
        client = MongoClient("localhost", 27017)
        database = client.get_database("test_remiss")
        collection = database.get_collection('test_tweets')
        df = pd.DataFrame(list(collection.find()))
        expected = df.groupby(pd.Grouper(key='created_at', freq='1D')).size()
        actual = load_tweet_count_evolution('localhost', 27017, 'test_remiss', 'test_tweets',
                                            unit='day', bin_size=1,
                                            start_date=start_date, end_date=end_date)

        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_load_tweet_count_evolution_start_end_date_2(self):
        start_date = pd.to_datetime('2019-01-01 23:20:00')
        end_date = pd.to_datetime('2020-12-30 23:59:59')
        client = MongoClient("localhost", 27017)
        database = client.get_database("test_remiss")
        collection = database.get_collection('test_tweets')
        df = pd.DataFrame(list(collection.find()))
        df = df[df['created_at'] < end_date]
        expected = df.groupby(pd.Grouper(key='created_at', freq='1D')).size()
        actual = load_tweet_count_evolution('localhost', 27017, 'test_remiss', 'test_tweets',
                                            unit='day', bin_size=1,
                                            start_date=start_date, end_date=end_date)

        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_load_tweet_count_evolution_hashtag(self):
        hashtag = 'CataluñaPorEspaña'
        client = MongoClient("localhost", 27017)
        database = client.get_database("test_remiss")
        collection = database.get_collection('test_tweets')
        df = pd.DataFrame(list(collection.find()))
        df = df[df['entities'].apply(lambda x: x['hashtags'][0]['tag'] == hashtag if len(x['hashtags']) else False)]
        expected = df.groupby(pd.Grouper(key='created_at', freq='1D')).size()
        expected = expected[expected > 0]
        actual = load_tweet_count_evolution('localhost', 27017, 'test_remiss', 'test_tweets',
                                            hashtag=hashtag,
                                            unit='day', bin_size=1)
        self.maxDiff = None
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_update_graph(self):
        fig = update_graph('test_tweets', start_date=None, end_date=None, hashtag=None)
        fig.show()

    def test_update_graph_hashtag(self):
        fig = update_graph('test_tweets', start_date=None, end_date=None, hashtag=['CataluñaPorEspaña', 1])
        fig.show()

    def test_update_graph_date_range(self):
        start_date = pd.to_datetime('2019-01-01 23:20:00')
        end_date = pd.to_datetime('2020-01-1 23:59:59')
        fig = update_graph('test_tweets', start_date=start_date, end_date=end_date, hashtag=None)
        fig.show()

    def test_update_graph_date_range_hashtag(self):
        start_date = pd.to_datetime('2019-01-01 23:20:00')
        end_date = pd.to_datetime('2020-01-1 23:59:59')
        fig = update_graph('test_tweets', start_date=start_date, end_date=end_date,
                           hashtag=['CataluñaPorEspaña', 1])
        fig.show()

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
        output_file = 'test_resources/test.jsonl'
        generate_test_data(start_date, end_date, hashtags, parties, num_tweets, prob_hashtag, prob_party,
                           prob_usual_suspects, output_file)
