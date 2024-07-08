import pandas as pd
import plotly.express as px
import sklearn
from pymongo import MongoClient
from pymongoarrow.schema import Schema
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import igraph as ig
import pymongoarrow

pymongoarrow.monkey.patch_all()

sklearn.set_config(transform_output="pandas")


class PropagationDatasetGenerator:
    def __init__(self, dataset, host='localhost', port=27017, reference_types=('replied_to', 'quoted', 'retweeted'),
                 use_profiling='full', use_textual='full'):
        self.host = host
        self.port = port
        self.reference_types = reference_types
        self.use_profiling = use_profiling
        self.use_textual = use_textual
        self._features = None
        self.dataset = dataset

    def generate_propagation_dataset(self):
        if self._features is None:
            self._features = self._generate_propagation_dataset()
        return self._features

    def get_cascade(self, conversation_id, user_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)

        collection = database.get_collection('raw')
        pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {
                'referenced_tweets.type': {'$in': self.reference_types},
                'referenced_tweets.author': {'$exists': True}
            }},
            {'$project': {
                '_id': 0,
                'source': '$referenced_tweets.author.id',
                'target': '$author.id',
            }}
        ]
        edges = collection.aggregate_pandas_all(pipeline)
        nested_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$project': {'author_id': '$author.id',
                          'username': '$author.username',
                          'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                          'party': '$author.remiss_metadata.party',
                          }
             }]

        node_pipeline = [
            {'$match': {'conversation_id': conversation_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {'referenced_tweets.type': {'$in': self.reference_types},
                        'referenced_tweets.author': {'$exists': True}}},
            {'$project': {'_id': 0, 'author_id': '$referenced_tweets.author.id',
                          'username': '$referenced_tweets.author.username',
                          'is_usual_suspect': '$referenced_tweets.author.remiss_metadata.is_usual_suspect',
                          'party': '$referenced_tweets.author.remiss_metadata.party',
                          }},
            {'$unionWith': {'coll': 'raw', 'pipeline': nested_pipeline}},  # Fetch missing authors
            {'$group': {'_id': '$author_id',
                        'username': {'$first': '$username'},
                        'is_usual_suspect': {'$addToSet': '$is_usual_suspect'},
                        'party': {'$addToSet': '$party'},
                        }},

            {'$project': {'_id': 0,
                          'author_id': '$_id',
                          'username': 1,
                          'is_usual_suspect': {'$anyElementTrue': '$is_usual_suspect'},
                          'party': {'$arrayElemAt': ['$party', 0]},
                          }},

        ]
        schema = Schema({'author_id': str, 'username': str, 'is_usual_suspect': bool, 'party': str})
        authors = collection.aggregate_pandas_all(node_pipeline, schema=schema).drop_duplicates(subset='author_id')
        client.close()

        graph = ig.Graph(directed=True)
        graph.add_vertices(len(authors))
        graph.vs['username'] = authors['username']
        graph.vs['author_id'] = authors['author_id']
        graph.vs['is_usual_suspect'] = authors['is_usual_suspect']
        graph.vs['party'] = authors['party']
        graph.vs['original'] = ['ground_truth'] * len(authors)
        if user_id is not None:
            graph.vs.find(author_id=user_id)['original'] = 'current_user'

        if not edges.empty:
            author_to_id = authors['author_id'].reset_index().set_index('author_id')

            edges['source'] = author_to_id.loc[edges['source']].reset_index(drop=True)
            edges['target'] = author_to_id.loc[edges['target']].reset_index(drop=True)

            graph.add_edges(edges[['source', 'target']].to_records(index=False).tolist())

        graph['conversation_id'] = conversation_id

        return graph

    def get_neighbours(self, user_id):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)

        collection = database.get_collection('raw')
        pipeline = [
            {'$match': {'author.id': user_id}},
            {'$unwind': '$referenced_tweets'},
            {'$match': {
                'referenced_tweets.type': {'$in': self.reference_types},
                'referenced_tweets.author': {'$exists': True}
            }},
            {'$project': {
                '_id': 0,
                'author_id': '$referenced_tweets.author.id'
            }}
        ]
        neighbours = collection.aggregate_pandas_all(pipeline)['author_id'].unique()
        client.close()
        return neighbours

    def get_features_for(self, conversation_id, source, target):
        client = MongoClient(self.host, self.port)
        database = client.get_database(self.dataset)

        tweet_features = self._fetch_tweet_features(database)
        user_features = self._fetch_user_features(database)
        edges = pd.DataFrame({'source': [source], 'target': [target], 'conversation_id': [conversation_id]})
        features = self._merge_features(edges, tweet_features, user_features)
        features = features.drop(columns=['source', 'target', 'conversation_id', 'Unnamed: 0'], errors='ignore')
        client.close()

        return features

    def _generate_propagation_dataset(self):
        client = MongoClient(self.host, self.port)

        database = client.get_database(self.dataset)

        tweet_features = self._fetch_tweet_features(database)
        user_features = self._fetch_user_features(database)
        edges = self._fetch_edges(database)

        client.close()

        features = self._merge_features(edges, tweet_features, user_features)
        negatives = self._generate_negative_samples(edges, tweet_features, user_features)

        features['propagated'] = 1
        negatives['propagated'] = 0

        dataset = pd.concat([features, negatives]).reset_index(drop=True)
        dataset = dataset.drop(columns=['source', 'target', 'conversation_id', 'Unnamed: 0'], errors='ignore')

        # Shuffle just in case
        dataset = dataset.sample(frac=1).reset_index(drop=True)

        print('Features generated')
        print(f'Num positives: {len(features)}')
        print(f'Num negatives: {len(negatives)}')

        return dataset

    def _fetch_tweet_features(self, database):
        features = self._fetch_raw_tweet_features(database)
        if self.use_textual in {'full', 'strict'}:
            how = 'inner' if self.use_textual == 'strict' else 'left'
            textual_features = self._fetch_textual_tweet_features(database)
            return features.merge(textual_features, left_index=True, right_index=True, how=how)
        elif not self.use_textual:
            return features
        else:
            raise ValueError(f'Invalid value for use_textual: {self.use_textual}')

    def _fetch_raw_tweet_features(self, database):
        collection = database.get_collection('raw')
        pipeline = [
            {'$match': {'$expr': {'$eq': ['$id', '$conversation_id']}}},
            {'$project': {
                '_id': 0,
                'tweet_id': '$conversation_id',
                'num_hashtags': {'$size': {'$ifNull': ['$entities.hashtags', []]}},
                'num_mentions': {'$size': {'$ifNull': ['$entities.mentions', []]}},
                'num_urls': {'$size': {'$ifNull': ['$entities.urls', []]}},
                'num_media': {'$size': {'$ifNull': ['$entities.media', []]}},
                'num_interactions': {'$size': {'$ifNull': ['$referenced_tweets', []]}},
                'num_words': {'$size': {'$split': ['$text', ' ']}},
                'num_chars': {'$strLenCP': '$text'},
                'is_usual_suspect_op': '$author.remiss_metadata.is_usual_suspect',
                'party_op': '$author.remiss_metadata.party'
            }}
        ]
        return collection.aggregate_pandas_all(pipeline).set_index('tweet_id')

    def _fetch_textual_tweet_features(self, database):
        collection = database.get_collection('textual')
        pipeline = [
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

        return features.set_index('tweet_id')

    def _fetch_propagation_metrics(self, database):
        collection = database.get_collection('user_propagation')
        pipeline = [
            {'$project': {'_id': 0, 'author_id': 1, 'legitimacy': 1, 't-closeness': 1}}
        ]
        propagation_metrics = collection.aggregate_pandas_all(pipeline)
        if propagation_metrics.empty:
            raise RuntimeError('Propagation metrics not found. Please prepopulate them first')
        return propagation_metrics.set_index('author_id')

    def _fetch_user_features(self, database):
        propagation_metrics = self._fetch_propagation_metrics(database)
        features = self._fetch_raw_user_features(database)
        features = features.merge(propagation_metrics, left_index=True, right_index=True, how='inner')

        if self.use_profiling in {'full', 'strict'}:
            profiling_features = self._fetch_profiling_user_features(database)
            merge_how = 'inner' if self.use_profiling == 'strict' else 'left'
            return features.merge(profiling_features, left_index=True, right_index=True, how=merge_how)
        elif not self.use_profiling:
            return features

        else:
            raise ValueError(f'Invalid value for use_profiling: {self.use_profiling}')

    def _fetch_raw_user_features(self, database):
        collection = database.get_collection('raw')
        pipeline = [
            {'$project': {
                '_id': 0,
                'author_id': '$author.id',
                'is_usual_suspect': '$author.remiss_metadata.is_usual_suspect',
                'party': '$author.remiss_metadata.party'
            }}
        ]
        user_features = collection.aggregate_pandas_all(pipeline).drop_duplicates(subset='author_id')
        return user_features.set_index('author_id')

    def _fetch_profiling_user_features(self, database):
        collection = database.get_collection('profiling')

        pipeline = [
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

    def _fetch_edges(self, database):
        collection = database.get_collection('raw')
        pipeline = [
            {'$unwind': '$referenced_tweets'},
            {'$match': {
                'referenced_tweets.type': {'$in': self.reference_types},
                'referenced_tweets.author': {'$exists': True}
            }},
            {'$project': {
                '_id': 0,
                'source': '$referenced_tweets.author.id',
                'target': '$author.id',
                'conversation_id': '$conversation_id'
            }}
        ]
        return collection.aggregate_pandas_all(pipeline)

    def _merge_features(self, edges, tweet_features, user_features):
        features = edges.merge(tweet_features, left_on='conversation_id', right_index=True, how='left')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_prev'), left_on='source',
                                  right_index=True, how='left')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_curr'), left_on='target',
                                  right_index=True, how='left')
        return features

    def _generate_negative_samples(self, edges, tweet_features, user_features):
        negatives = []
        for source, interactions in edges.groupby('source'):
            if len(interactions) > 1:
                targets = set(interactions['target'].unique())
                for conversation_id, conversation in interactions.groupby('conversation_id'):
                    other_targets = pd.DataFrame(targets - set(conversation['target']), columns=['target'])
                    if not other_targets.empty:
                        sample_size = min(len(conversation), len(other_targets))
                        other_targets = other_targets.sample(n=sample_size)
                        other_targets['source'] = source
                        other_targets['conversation_id'] = conversation_id
                        negatives.append(other_targets)

        if negatives:
            negatives = pd.concat(negatives)
            negatives = negatives.merge(tweet_features, left_on='conversation_id', right_index=True, how='inner')
            negatives = negatives.merge(user_features.rename(columns=lambda x: f'{x}_prev'), left_on='source',
                                        right_index=True, how='inner')
            negatives = negatives.merge(user_features.rename(columns=lambda x: f'{x}_curr'), left_on='target',
                                        right_index=True, how='inner')

        return negatives


class PropagationCascadeModel(BaseEstimator, ClassifierMixin):
    def __init__(self, host='localhost', port=27017, reference_types=('replied_to', 'quoted', 'retweeted'),
                 use_profiling='full', use_textual='full', dataset_generator=None):
        self.pipeline = None
        self.host = host
        self.port = port
        self.reference_types = reference_types
        self.use_profiling = use_profiling
        self.use_textual = use_textual
        self.dataset_generator = dataset_generator

    def fit(self, X, y=None):
        if isinstance(X, str):
            dataset = X
            generator = PropagationDatasetGenerator(dataset, host=self.host, port=self.port,
                                                    reference_types=self.reference_types,
                                                    use_profiling=self.use_profiling, use_textual=self.use_textual)
            X = generator.generate_propagation_dataset()
            self.dataset_generator = generator
        if y is None:
            y = X['propagated']
            X = X.drop(columns=['propagated'])
        self.pipeline = self._fit_propagation_model(X, y)

    def _fit_propagation_model(self, X, y):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('transformer', ColumnTransformer([
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 X.select_dtypes(include='object').columns),
            ], remainder='passthrough', verbose_feature_names_out=False)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier())
        ])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        print('Training set metrics')
        print(classification_report(y_train, y_train_pred))
        print('Confusion matrix')
        print(pd.crosstab(y_train, y_train_pred, rownames=['Actual'], colnames=['Predicted']))
        print('Test set metrics')
        print(classification_report(y_test, y_test_pred))
        print('Confusion matrix')
        print(pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predicted']))
        # plot feature importance
        feature_importances = pd.Series(pipeline['classifier'].feature_importances_,
                                        index=pipeline['classifier'].feature_names_in_).sort_values(ascending=False)
        fig = px.bar(feature_importances, title='Feature importance')
        fig.update_xaxes(title_text='Feature')
        fig.update_yaxes(title_text='Importance')
        fig.show()
        return pipeline

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError('Model not fitted')
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if self.pipeline is None:
            raise RuntimeError('Model not fitted')
        return self.pipeline.predict_proba(X)

    def score(self, X, y=None, **kwargs):
        if self.pipeline is None:
            raise RuntimeError('Model not fitted')
        return self.pipeline.score(X, y)

    def generate_cascade(self, x):
        # Get the conversation id and user id from x
        conversation_id = x['conversation_id']
        user_id = x['author_id']
        # Get the cascade of a given tweet
        cascade = self.dataset_generator.get_cascade(conversation_id, user_id)
        cascade.vs['original'] = [True] * len(cascade.vs)
        self._process_neighbour_propagation(x['author_id'], cascade)
        return cascade

    def _process_neighbour_propagation(self, author_id, cascade):
        for neighbour in self.dataset_generator.get_neighbours(author_id):
            # get features for neighbour
            neighbour_features = self.dataset_generator.get_features_for(cascade['conversation_id'], author_id,
                                                                        neighbour)
            # get the prediction on whether the neighbour will propagate the tweet
            prediction = self.predict(neighbour_features)
            # add the neighbour to the cascade
            if prediction:
                cascade.add_vertex(neighbour)
                cascade.add_edge(author_id, neighbour)
                cascade.vs['original'] += [False]
                # if propagation is predicted, recursively process the next neighbours
                self._process_neighbour_propagation(neighbour, cascade)

        return cascade
