import logging
import random
import time

import pandas as pd
import plotly.graph_objects as go
import sklearn
from joblib import Parallel, delayed
from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
from pymongoarrow.schema import Schema
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm
from xgboost import XGBClassifier

from propagation import DiffusionMetrics, Egonet

patch_all()

sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)


class PropagationDatasetGenerator:
    def __init__(self, dataset, host='localhost', port=27017, num_samples=None):
        self.num_samples = num_samples
        self.host = host
        self.port = port
        self._features = None
        self.dataset = dataset
        self.user_features = {}
        self.tweet_features = {}
        self.egonet = Egonet(host, port)
        self.diffusion_metrics = DiffusionMetrics(host=host, port=port, egonet=self.egonet)

    def get_available_cascades(self):
        logger.info('Fetching available cascades')
        profiling_author_ids = self._get_profiling_author_ids()
        cascades = self._get_cascades(profiling_author_ids)
        cascades = self._filter_textual(cascades)
        if cascades.empty:
            raise RuntimeError('No cascades found. Please prepopulate them first')

        logger.info(f'Found {len(cascades)} cascades')
        return cascades.reset_index(drop=True)

    def _get_profiling_author_ids(self):
        logger.info('Fetching profiling author ids')
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('profiling')
        author_ids = collection.distinct('user_id')
        client.close()
        logger.info(f'Found {len(author_ids)} profiling author ids')
        return list(author_ids)

    def _get_cascades(self, profiling_author_ids):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('raw')
        original_tweet_pipeline = [
            {'$match': {'author.id': {'$in': profiling_author_ids}, 'referenced_tweets': {'$exists': False}}},
            {'$project': {
                '_id': 0,
                'tweet_id': '$id',
                'author_id': '$author.id',
                'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                'party': '$author.remiss_metadata.party'
            }}
        ]
        schema = Schema({
            'tweet_id': str,
            'author_id': str,
            'is_usual_suspect': bool,
            'party': str
        })
        original_tweets = collection.aggregate_pandas_all(original_tweet_pipeline, schema=schema)
        propagated_tweet_pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.id': {'$in': original_tweets['tweet_id'].tolist()}}},
            {'$project': {'tweet_id': '$referenced_tweets.id', }}
        ]
        propagated_tweets = collection.aggregate_pandas_all(propagated_tweet_pipeline)
        client.close()
        cascades = original_tweets[original_tweets['tweet_id'].isin(propagated_tweets['tweet_id'])]
        cascades = cascades.drop_duplicates(subset='tweet_id')

        return cascades

    def _filter_textual(self, cascades):
        # remove cascades without textual features
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('textual')
        tweet_ids = cascades['tweet_id'].astype(int).tolist()
        pipeline = [
            {'$match': {'id': {'$in': tweet_ids}}},
            {'$project': {'_id': 0, 'tweet_id': '$id'}}
        ]
        textual_tweets = collection.aggregate_pandas_all(pipeline)
        client.close()

        if textual_tweets.empty:
           logger.warning('No cascades with textual features found, picking them all')
        else:
            textual_tweets['tweet_id'] = textual_tweets['tweet_id'].astype(int).astype(str)
            textual_tweets = textual_tweets.drop_duplicates(subset='tweet_id')
            good = cascades['tweet_id'].isin(textual_tweets['tweet_id'])
            cascades = cascades[good]
        return cascades

    def get_rows(self, cascades):
        logger.info(f'Fetching rows for {len(cascades)} cascades')

        if self.num_samples is not None:
            max_samples_per_cascade = self.num_samples // len(cascades)
        else:
            max_samples_per_cascade = None
        jobs = []
        for _, cascade in cascades.iterrows():
            jobs.append(delayed(self._get_row_for_cascades)(cascade, max_samples_per_cascade))
        rows = Parallel(n_jobs=-2, verbose=10)(jobs)
        rows = pd.DataFrame([row for sublist in rows for row in sublist])
        logger.info(f'Found {len(rows)} rows')
        return rows

    def _get_row_for_cascades(self, cascade, max_samples_per_cascade=None):
        rows = []
        prop_tree = self._get_propagation_tree(cascade['tweet_id'])

        for node in random.random.shuffle(prop_tree.vs):
            neighbors = set(prop_tree.neighbors(node))
            for neighbor in neighbors:
                if neighbor != node.index:
                    neighbor = prop_tree.vs[neighbor]
                    rows.append({
                        'source': node['author_id'],
                        'target': neighbor['author_id'],
                        'cascade_id': cascade['tweet_id'],
                        'original_author': cascade['author_id'],
                    })
        return rows

    def _get_propagation_tree(self, tweet_id):
        try:
            prop_tree = self.diffusion_metrics.get_propagation_tree(self.dataset, tweet_id)
        except RuntimeError:
            prop_tree = self.diffusion_metrics.compute_propagation_tree(self.dataset, tweet_id)
        return prop_tree

    def fetch_tweet_features(self, tweets):
        raw_features = self._fetch_raw_tweet_features(tweets)
        textual_features = self._fetch_textual_tweet_features(tweets)
        return raw_features.join(textual_features, how='left')

    def _fetch_raw_tweet_features(self, tweets):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('raw')

        pipeline = [
            {'$match': {'id': {'$in': tweets}}},
            {'$project': {'_id': 0, 'tweet_id': '$id',
                            'num_replies': '$public_metrics.reply_count', 'num_retweets': '$public_metrics.retweet_count',
                            'num_quotes': '$public_metrics.quote_count', 'num_likes': '$public_metrics.like_count',
                          }}
        ]
        raw_tweet_features = collection.aggregate_pandas_all(pipeline)
        if raw_tweet_features.empty:
            raise RuntimeError('Tweet features not found. Please prepopulate them first')
        return raw_tweet_features.set_index('tweet_id')

    def _fetch_textual_tweet_features(self, tweets):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('textual')
        tweets = [int(tweet) for tweet in tweets]
        pipeline = [
            {'$match': {'id': {'$in': tweets}}},
            {'$project': {
                '_id': 0,
                'tweet_id': '$id',
                'language': 1,
                'possibly_sensitive': 1,
                'retweet_count': 1,
                'reply_count': 1,
                'like_count': 1,
                'sentences': 1,
                'POS_entities_1d': 1,
                'POS_tags_1d': 1,
                'TFIDF_1d': 1,
                'No ironico': 1,
                'Ironia': 1,
                'Odio': 1,
                'Dirigido': 1,
                'Agresividad': 1,
                'others': 1,
                'Diversion': 1,
                'Tristeza': 1,
                'Enfado': 1,
                'Sorpresa': 1,
                'Disgusto': 1,
                'Miedo': 1,
                'Negativo': 1,
                'Neutro': 1,
                'Positivo': 1,
                'REAL': 1,
                'FAKE': 1,
                'Toxico': 1,
                'Muy toxico': 1,
                'fakeness': 1,
                'fakeness_probabilities': 1,
            }}
        ]
        features = collection.aggregate_pandas_all(pipeline)

        if features.empty:
            logger.warning('Textual features not found. Please prepopulate them first')
        else:
            features['tweet_id'] = features['tweet_id'].astype(str)
            features = features.set_index('tweet_id')
        return features

    def fetch_user_features(self, users):
        raw_features = self._fetch_raw_user_features(users)
        profiling_features = self._fetch_profiling_user_features(users)
        network_features = self._fetch_network_metrics(users)
        return raw_features.join(profiling_features, how='left').join(network_features, how='left')

    def _fetch_network_metrics(self, users):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('network_metrics')
        pipeline = [
            {'$match': {'author_id': {'$in': users}}},
            {'$project': {'_id': 0, 'author_id': 1,
                          'legitimacy': 1,
                          'reputation': '$average_reputation',
                          'status': '$average_status',
                          }}
        ]
        network_metrics = collection.aggregate_pandas_all(pipeline)
        if network_metrics.empty:
            raise RuntimeError('network metrics not found. Please prepopulate them first')
        return network_metrics.set_index('author_id')

    def _fetch_raw_user_features(self, users):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('raw')
        pipeline = [
            {'$match': {'author.id': {'$in': users}}},
            {'$project': {
                '_id': 0,
                'author_id': '$author.id',
                'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                'party': '$author.remiss_metadata.party'
            }}
        ]
        user_features = collection.aggregate_pandas_all(pipeline).drop_duplicates(subset='author_id')
        return user_features.set_index('author_id')

    def _fetch_profiling_user_features(self, users):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)
        collection = database.get_collection('profiling')

        pipeline = [
            {'$match': {'user_id': {'$in': users}}},
            {'$project':
                 {'_id': 0, 'author_id': '$user_id', 'is_verified': 1,
                  'followers_count': 1, 'friends_count': 1, 'listed_count': 1, 'favorites_count': 1,
                  'statuses_count': 1, 'url_in_profile': 1,
                  'help_empath': 1, 'office_empath': 1, 'dance_empath': 1, 'money_empath': 1,
                  'wedding_empath': 1, 'domestic_work_empath': 1, 'sleep_empath': 1, 'medical_emergency_empath': 1,
                  'cold_empath': 1, 'hate_empath': 1, 'cheerfulness_empath': 1, 'aggression_empath': 1,
                  'occupation_empath': 1, 'envy_empath': 1, 'anticipation_empath': 1, 'family_empath': 1,
                  'vacation_empath': 1, 'crime_empath': 1, 'attractive_empath': 1, 'masculine_empath': 1,
                  'prison_empath': 1, 'health_empath': 1, 'pride_empath': 1, 'dispute_empath': 1,
                  'nervousness_empath': 1, 'government_empath': 1, 'weakness_empath': 1, 'horror_empath': 1,
                  'swearing_terms_empath': 1, 'leisure_empath': 1, 'suffering_empath': 1, 'royalty_empath': 1,
                  'wealthy_empath': 1, 'tourism_empath': 1, 'furniture_empath': 1, 'school_empath': 1,
                  'magic_empath': 1, 'beach_empath': 1, 'journalism_empath': 1, 'morning_empath': 1,
                  'banking_empath': 1, 'social_media_empath': 1, 'exercise_empath': 1, 'night_empath': 1,
                  'kill_empath': 1, 'blue_collar_job_empath': 1, 'art_empath': 1, 'ridicule_empath': 1,
                  'play_empath': 1, 'computer_empath': 1, 'college_empath': 1, 'optimism_empath': 1,
                  'stealing_empath': 1, 'real_estate_empath': 1, 'home_empath': 1, 'divine_empath': 1,
                  'sexual_empath': 1, 'fear_empath': 1, 'irritability_empath': 1, 'superhero_empath': 1,
                  'business_empath': 1, 'driving_empath': 1, 'pet_empath': 1, 'childish_empath': 1, 'cooking_empath': 1,
                  'exasperation_empath': 1, 'religion_empath': 1, 'hipster_empath': 1, 'internet_empath': 1,
                  'surprise_empath': 1, 'reading_empath': 1, 'worship_empath': 1, 'leader_empath': 1,
                  'independence_empath': 1, 'movement_empath': 1, 'body_empath': 1, 'noise_empath': 1,
                  'eating_empath': 1, 'medieval_empath': 1, 'zest_empath': 1, 'confusion_empath': 1, 'water_empath': 1,
                  'sports_empath': 1, 'death_empath': 1, 'healing_empath': 1, 'legend_empath': 1, 'heroic_empath': 1,
                  'celebration_empath': 1, 'restaurant_empath': 1, 'violence_empath': 1, 'programming_empath': 1,
                  'dominant_heirarchical_empath': 1, 'military_empath': 1, 'neglect_empath': 1, 'swimming_empath': 1,
                  'exotic_empath': 1, 'love_empath': 1, 'hiking_empath': 1, 'communication_empath': 1,
                  'hearing_empath': 1, 'order_empath': 1, 'sympathy_empath': 1, 'hygiene_empath': 1,
                  'weather_empath': 1, 'anonymity_empath': 1, 'trust_empath': 1, 'ancient_empath': 1,
                  'deception_empath': 1, 'fabric_empath': 1, 'air_travel_empath': 1, 'fight_empath': 1,
                  'dominant_personality_empath': 1, 'music_empath': 1, 'vehicle_empath': 1, 'politeness_empath': 1,
                  'toy_empath': 1, 'farming_empath': 1, 'meeting_empath': 1, 'war_empath': 1, 'speaking_empath': 1,
                  'listen_empath': 1, 'urban_empath': 1, 'shopping_empath': 1, 'disgust_empath': 1, 'fire_empath': 1,
                  'tool_empath': 1, 'phone_empath': 1, 'gain_empath': 1, 'sound_empath': 1, 'injury_empath': 1,
                  'sailing_empath': 1, 'rage_empath': 1, 'science_empath': 1, 'work_empath': 1, 'appearance_empath': 1,
                  'valuable_empath': 1, 'warmth_empath': 1, 'youth_empath': 1, 'sadness_empath': 1, 'fun_empath': 1,
                  'emotional_empath': 1, 'joy_empath': 1, 'affection_empath': 1, 'traveling_empath': 1,
                  'fashion_empath': 1, 'ugliness_empath': 1, 'lust_empath': 1, 'shame_empath': 1, 'torment_empath': 1,
                  'economics_empath': 1, 'anger_empath': 1, 'politics_empath': 1, 'ship_empath': 1,
                  'clothing_empath': 1, 'car_empath': 1, 'strength_empath': 1, 'technology_empath': 1,
                  'breaking_empath': 1, 'shape_and_size_empath': 1, 'power_empath': 1, 'white_collar_job_empath': 1,
                  'animal_empath': 1, 'party_empath': 1, 'terrorism_empath': 1, 'smell_empath': 1,
                  'disappointment_empath': 1, 'poor_empath': 1, 'plant_empath': 1, 'pain_empath': 1, 'beauty_empath': 1,
                  'timidity_empath': 1, 'philosophy_empath': 1, 'negotiate_empath': 1, 'negative_emotion_empath': 1,
                  'cleaning_empath': 1, 'messaging_empath': 1, 'competing_empath': 1, 'law_empath': 1,
                  'friends_empath': 1, 'payment_empath': 1, 'achievement_empath': 1, 'alcohol_empath': 1,
                  'liquid_empath': 1, 'feminine_empath': 1, 'weapon_empath': 1, 'children_empath': 1,
                  'monster_empath': 1, 'ocean_empath': 1, 'giving_empath': 1, 'contentment_empath': 1,
                  'writing_empath': 1, 'rural_empath': 1, 'positive_emotion_empath': 1, 'musical_empath': 1,
                  'anticipation_emolex': 1, 'surprise_emolex': 1, 'joy_emolex': 1, 'positive_emolex': 1,
                  'anger_emolex': 1, 'trust_emolex': 1, 'disgust_emolex': 1, 'fear_emolex': 1, 'sadness_emolex': 1,
                  'negative_emolex': 1, 'negative_sentiment': 1, 'neutral_sentiment': 1, 'positive_sentiment': 1,
                  'NOT-HATE_hate_sp': 1, 'hate_hate_sp': 1, 'Verbos_liwc': 1, 'Present_liwc': 1, 'verbosEL_liwc': 1,
                  'Funct_liwc': 1, 'TotPron_liwc': 1, 'PronImp_liwc': 1, 'MecCog_liwc': 1, 'Tentat_liwc': 1,
                  'Conjunc_liwc': 1, 'Causa_liwc': 1, 'VerbAux_liwc': 1, 'Social_liwc': 1, 'Afect_liwc': 1,
                  'EmoPos_liwc': 1, 'Certeza_liwc': 1, 'Prepos_liwc': 1, 'Biolog_liwc': 1, 'Cuerpo_liwc': 1,
                  'Incl_liwc': 1, 'Negacio_liwc': 1, 'Sexual_liwc': 1, 'Adverb_liwc': 1, 'Relativ_liwc': 1,
                  'Tiempo_liwc': 1, 'Insight_liwc': 1, 'Discrep_liwc': 1, 'Salud_liwc': 1, 'Excl_liwc': 1,
                  'gender_demo': 1, 'age_demo': 1, 'retweets_count_social': 1, 'favs_count_social': 1,
                  'mentions_count_social': 1, 'ratio_quoted_tweet_types': 1, 'ratio_retweets_tweet_types': 1,
                  'ratio_replies_tweet_types': 1, 'ratio_original_tweet_types': 1, 'week_days_count_ratio_behav': 1,
                  'weekend_days_count_ratio_behav': 1, 'median_time_betweet_tweets_behav': 1,
                  'tweets_sleep_time_ratio_behav': 1, 'tweets_awake_time_ratio_behav': 1, 'lang': 1,
                  }
             }
        ]
        user_features = collection.aggregate_pandas_all(pipeline).set_index('author_id')
        return user_features

    def _generate_negative_rows(self, features):
        logger.info('Generating negative samples')
        hidden_network = self.egonet.get_hidden_network(self.dataset)
        negatives = []
        for source, interactions in tqdm(features.groupby('source')):
            if len(interactions) > 1:
                targets = set(interactions['target'])
                source_node = hidden_network.vs.find(author_id=source)
                neighbors = hidden_network.neighbors(source_node)
                neighbors = set(neighbors) - targets
                for target in neighbors:
                    neighbor = hidden_network.vs[target]
                    negatives.append({
                        'source': source,
                        'target': neighbor['author_id'],
                        'cascade_id': interactions['cascade_id'].iloc[0],
                        'original_author': interactions['original_author'].iloc[0],
                    })

        if not negatives:
            raise RuntimeError('No negative samples found')
        negatives = pd.DataFrame(negatives).astype(str)
        return negatives

    def generate_propagation_dataset(self):
        logger.info('Generating propagation dataset')
        cascades = self.get_available_cascades()

        positive_rows = self.get_rows(cascades)

        tweets = positive_rows['cascade_id'].unique().tolist()
        logger.info(f'Fetching tweet features for {len(tweets)} tweets')
        positive_users = list(set(positive_rows['source'].unique()) | set(positive_rows['target'].unique()))
        logger.info(f'Fetching user features for {len(positive_users)} users')
        tweet_features = self.fetch_tweet_features(tweets)
        logger.info(f'Fetched {len(tweet_features)} tweet features')
        positive_user_features = self.fetch_user_features(positive_users)
        logger.info(f'Fetched {len(positive_user_features)} user features')


        logger.info('Merging features')
        positive_samples = positive_rows.merge(tweet_features, left_on='cascade_id', right_index=True, how='inner')
        positive_samples = positive_samples.merge(positive_user_features.rename(columns=lambda x: f'{x}_prev'),
                                                  left_on='source',
                                                  right_index=True, how='inner')
        positive_samples = positive_samples.merge(positive_user_features.rename(columns=lambda x: f'{x}_curr'),
                                                  left_on='target',
                                                  right_index=True, how='inner')
        positive_samples = positive_samples.merge(positive_user_features.rename(columns=lambda x: f'{x}_original'),
                                                  left_on='original_author',
                                                  right_index=True, how='inner')

        logger.info('Generating negative samples')
        negative_rows = self._generate_negative_rows(positive_samples)
        negative_users = list(set(negative_rows['source'].unique()) | set(negative_rows['target'].unique()))
        logger.info(f'Fetching user features for {len(negative_users)} users')
        negative_user_features = self.fetch_user_features(negative_users)
        logger.info(f'Fetched {len(negative_user_features)} user features')

        logger.info('Merging negative features')
        negative_samples = negative_rows.merge(tweet_features, left_on='cascade_id', right_index=True, how='inner')
        negative_samples = negative_samples.merge(negative_user_features.rename(columns=lambda x: f'{x}_prev'),
                                                  left_on='source',
                                                  right_index=True, how='inner')
        negative_samples = negative_samples.merge(negative_user_features.rename(columns=lambda x: f'{x}_curr'),
                                                  left_on='target',
                                                  right_index=True, how='inner')
        negative_samples = negative_samples.merge(negative_user_features.rename(columns=lambda x: f'{x}_original'),
                                                  left_on='original_author',
                                                  right_index=True, how='inner')

        positive_samples['propagated'] = 1
        negative_samples['propagated'] = 0
        dataset = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        dataset = dataset.drop(columns=['source', 'target', 'cascade_id', 'original_author'], errors='ignore')
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        logger.info(f'Generated dataset with {len(dataset)} samples')
        return dataset


class PropagationCascadeModel:
    def __init__(self, host='localhost', port=27017, reference_types=('replied_to', 'quoted', 'retweeted'),
                 use_profiling='full', use_textual='full', dataset_generator=None):
        self.host = host
        self.port = port
        self.reference_types = reference_types
        self.use_profiling = use_profiling
        self.use_textual = use_textual
        self.dataset_generator = dataset_generator
        self.model = None

    def fit(self, X, y):
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('transformer', ColumnTransformer([
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 X.select_dtypes(include='object').columns),
            ], remainder='passthrough', verbose_feature_names_out=False)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(scale_pos_weight=(len(y) - y.sum()) / y.sum()))
        ])
        model.fit(X, y)

        self.model = model
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('Model not fitted')
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError('Model not fitted')
        return self.model.predict_proba(X)

    def score(self, X, y=None, **kwargs):
        if self.model is None:
            raise RuntimeError('Model not fitted')
        return self.model.score(X, y)

    def generate_cascade(self, x):
        # Get the conversation id and user id from x
        conversation_id = x['conversation_id']
        user_id = x['author_id']
        # Get the cascade of a given tweet
        cascade = self.dataset_generator.get_cascade(conversation_id, user_id)
        visited_nodes = set(cascade.vs['author_id'])
        self._process_neighbour_propagation(x['author_id'], cascade, visited_nodes)
        return cascade

    def _process_neighbour_propagation(self, author_id, cascade, visited_nodes):
        neighbours = self.dataset_generator.get_neighbours(author_id)
        available_neighbours = set(neighbours) - visited_nodes
        visited_nodes.update(available_neighbours)
        sources = [author_id] * len(available_neighbours)
        targets = list(available_neighbours)
        features = self.dataset_generator.get_features_for(cascade['conversation_id'], sources, targets)
        if not features.empty:
            predictions = self.predict(features)
            predictions = pd.Series(predictions, index=targets)
            author_index = cascade.vs.find(author_id=author_id).index
            for target, prediction in predictions[predictions == 1].items():
                if target not in cascade.vs['author_id']:
                    cascade.add_vertex(author_id=target, username=target, original='predicted')
                target_index = cascade.vs.find(author_id=target).index
                cascade.add_edge(author_index, target_index)
                self._process_neighbour_propagation(target, cascade, visited_nodes)
        return cascade

    def plot_cascade(self, cascade):
        layout = cascade.layout('kk', dim=3)
        layout = pd.DataFrame(layout.coords, columns=['x', 'y', 'z'])
        print('Computing plot for network')
        print(cascade.summary())
        start_time = time.time()
        edges = pd.DataFrame(cascade.get_edgelist(), columns=['source', 'target'])
        edge_positions = layout.iloc[edges.values.flatten()].reset_index(drop=True)
        nones = edge_positions[1::2].assign(x=None, y=None, z=None)
        edge_positions = pd.concat([edge_positions, nones]).sort_index().reset_index(drop=True)

        color_map = {'ground_truth': 'rgb(255, 234, 208)', 'seed': 'rgb(247, 111, 142)',
                     'predicted': 'rgb(111, 247, 142)'}
        original = pd.Series(cascade.vs['original'])
        color = original.map(color_map)
        size = original.map({'ground_truth': 10, 'seed': 15, 'predicted': 10})

        # metadata = pd.DataFrame({'is_usual_suspect': network.vs['is_usual_suspect'], 'party': network.vs['party']})

        edge_trace = go.Scatter3d(x=edge_positions['x'],
                                  y=edge_positions['y'],
                                  z=edge_positions['z'],
                                  mode='lines',
                                  line=dict(color='rgb(125,125,125)', width=1),
                                  hoverinfo='none',
                                  name='Interactions',
                                  showlegend=False
                                  )

        text = []
        for node in cascade.vs:
            node_text = f'Author: {node["username"]}'
            text.append(node_text)

        node_trace = go.Scatter3d(x=layout['x'],
                                  y=layout['y'],
                                  z=layout['z'],
                                  mode='markers',
                                  marker=dict(
                                      # symbol=markers,
                                      size=size,
                                      color=color,
                                      # coloscale set to $champagne: #ffead0ff;
                                      # to $bright-pink-crayola: #f76f8eff;
                                      # colorscale=[[0, 'rgb(255, 234, 208)'], [1, 'rgb(247, 111, 142)']],
                                      # colorbar=dict(thickness=20, title='Legitimacy'),
                                      line=dict(color='rgb(50,50,50)', width=0.5),
                                  ),
                                  text=text,
                                  hovertemplate='%{text}',
                                  name='',
                                  )

        axis = dict(showbackground=False,
                    showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    title=''
                    )

        layout = go.Layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            # margin=dict(
            #     t=100
            # ),
            hovermode='closest',

        )

        data = [edge_trace, node_trace]
        fig = go.Figure(data=data, layout=layout)

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
        fig.update_layout(scene_camera=camera)
        print(f'Plot computed in {time.time() - start_time} seconds')
        return fig
