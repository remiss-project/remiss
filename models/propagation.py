import logging
import random

import pandas as pd
import sklearn
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
        self.feature_names = ['num_replies', 'num_retweets', 'num_quotes', 'num_likes', 'language',
                              'possibly_sensitive', 'retweet_count', 'reply_count', 'like_count', 'sentences',
                              'POS_entities_1d', 'POS_tags_1d', 'TFIDF_1d', 'No ironico', 'Ironia', 'Odio', 'Dirigido',
                              'Agresividad', 'others', 'Diversion', 'Tristeza', 'Enfado', 'Sorpresa', 'Disgusto',
                              'Miedo', 'Negativo', 'Neutro', 'Positivo', 'REAL', 'FAKE', 'Toxico', 'Muy toxico',
                              'fakeness', 'fakeness_probabilities', 'is_usual_suspect_prev', 'is_verified_prev',
                              'followers_count_prev', 'friends_count_prev', 'listed_count_prev', 'favorites_count_prev',
                              'statuses_count_prev', 'url_in_profile_prev', 'help_empath_prev', 'office_empath_prev',
                              'dance_empath_prev', 'money_empath_prev', 'wedding_empath_prev',
                              'domestic_work_empath_prev', 'sleep_empath_prev', 'medical_emergency_empath_prev',
                              'cold_empath_prev', 'hate_empath_prev', 'cheerfulness_empath_prev',
                              'aggression_empath_prev', 'occupation_empath_prev', 'envy_empath_prev',
                              'anticipation_empath_prev', 'family_empath_prev', 'vacation_empath_prev',
                              'crime_empath_prev', 'attractive_empath_prev', 'masculine_empath_prev',
                              'prison_empath_prev', 'health_empath_prev', 'pride_empath_prev', 'dispute_empath_prev',
                              'nervousness_empath_prev', 'government_empath_prev', 'weakness_empath_prev',
                              'horror_empath_prev', 'swearing_terms_empath_prev', 'leisure_empath_prev',
                              'suffering_empath_prev', 'royalty_empath_prev', 'wealthy_empath_prev',
                              'tourism_empath_prev', 'furniture_empath_prev', 'school_empath_prev', 'magic_empath_prev',
                              'beach_empath_prev', 'journalism_empath_prev', 'morning_empath_prev',
                              'banking_empath_prev', 'social_media_empath_prev', 'exercise_empath_prev',
                              'night_empath_prev', 'kill_empath_prev', 'blue_collar_job_empath_prev', 'art_empath_prev',
                              'ridicule_empath_prev', 'play_empath_prev', 'computer_empath_prev', 'college_empath_prev',
                              'optimism_empath_prev', 'stealing_empath_prev', 'real_estate_empath_prev',
                              'home_empath_prev', 'divine_empath_prev', 'sexual_empath_prev', 'fear_empath_prev',
                              'irritability_empath_prev', 'superhero_empath_prev', 'business_empath_prev',
                              'driving_empath_prev', 'pet_empath_prev', 'childish_empath_prev', 'cooking_empath_prev',
                              'exasperation_empath_prev', 'religion_empath_prev', 'hipster_empath_prev',
                              'internet_empath_prev', 'surprise_empath_prev', 'reading_empath_prev',
                              'worship_empath_prev', 'leader_empath_prev', 'independence_empath_prev',
                              'movement_empath_prev', 'body_empath_prev', 'noise_empath_prev', 'eating_empath_prev',
                              'medieval_empath_prev', 'zest_empath_prev', 'confusion_empath_prev', 'water_empath_prev',
                              'sports_empath_prev', 'death_empath_prev', 'healing_empath_prev', 'legend_empath_prev',
                              'heroic_empath_prev', 'celebration_empath_prev', 'restaurant_empath_prev',
                              'violence_empath_prev', 'programming_empath_prev', 'dominant_heirarchical_empath_prev',
                              'military_empath_prev', 'neglect_empath_prev', 'swimming_empath_prev',
                              'exotic_empath_prev', 'love_empath_prev', 'hiking_empath_prev',
                              'communication_empath_prev', 'hearing_empath_prev', 'order_empath_prev',
                              'sympathy_empath_prev', 'hygiene_empath_prev', 'weather_empath_prev',
                              'anonymity_empath_prev', 'trust_empath_prev', 'ancient_empath_prev',
                              'deception_empath_prev', 'fabric_empath_prev', 'air_travel_empath_prev',
                              'fight_empath_prev', 'dominant_personality_empath_prev', 'music_empath_prev',
                              'vehicle_empath_prev', 'politeness_empath_prev', 'toy_empath_prev', 'farming_empath_prev',
                              'meeting_empath_prev', 'war_empath_prev', 'speaking_empath_prev', 'listen_empath_prev',
                              'urban_empath_prev', 'shopping_empath_prev', 'disgust_empath_prev', 'fire_empath_prev',
                              'tool_empath_prev', 'phone_empath_prev', 'gain_empath_prev', 'sound_empath_prev',
                              'injury_empath_prev', 'sailing_empath_prev', 'rage_empath_prev', 'science_empath_prev',
                              'work_empath_prev', 'appearance_empath_prev', 'valuable_empath_prev',
                              'warmth_empath_prev', 'youth_empath_prev', 'sadness_empath_prev', 'fun_empath_prev',
                              'emotional_empath_prev', 'joy_empath_prev', 'affection_empath_prev',
                              'traveling_empath_prev', 'fashion_empath_prev', 'ugliness_empath_prev',
                              'lust_empath_prev', 'shame_empath_prev', 'torment_empath_prev', 'economics_empath_prev',
                              'anger_empath_prev', 'politics_empath_prev', 'ship_empath_prev', 'clothing_empath_prev',
                              'car_empath_prev', 'strength_empath_prev', 'technology_empath_prev',
                              'breaking_empath_prev', 'shape_and_size_empath_prev', 'power_empath_prev',
                              'white_collar_job_empath_prev', 'animal_empath_prev', 'party_empath_prev',
                              'terrorism_empath_prev', 'smell_empath_prev', 'disappointment_empath_prev',
                              'poor_empath_prev', 'plant_empath_prev', 'pain_empath_prev', 'beauty_empath_prev',
                              'timidity_empath_prev', 'philosophy_empath_prev', 'negotiate_empath_prev',
                              'negative_emotion_empath_prev', 'cleaning_empath_prev', 'messaging_empath_prev',
                              'competing_empath_prev', 'law_empath_prev', 'friends_empath_prev', 'payment_empath_prev',
                              'achievement_empath_prev', 'alcohol_empath_prev', 'liquid_empath_prev',
                              'feminine_empath_prev', 'weapon_empath_prev', 'children_empath_prev',
                              'monster_empath_prev', 'ocean_empath_prev', 'giving_empath_prev',
                              'contentment_empath_prev', 'writing_empath_prev', 'rural_empath_prev',
                              'positive_emotion_empath_prev', 'musical_empath_prev', 'surprise_emolex_prev',
                              'trust_emolex_prev', 'positive_emolex_prev', 'anger_emolex_prev', 'disgust_emolex_prev',
                              'anticipation_emolex_prev', 'sadness_emolex_prev', 'joy_emolex_prev',
                              'negative_emolex_prev', 'fear_emolex_prev', 'negative_sentiment_prev',
                              'neutral_sentiment_prev', 'positive_sentiment_prev', 'NOT-HATE_hate_sp_prev',
                              'hate_hate_sp_prev', 'Verbos_liwc_prev', 'Present_liwc_prev', 'verbosEL_liwc_prev',
                              'Funct_liwc_prev', 'Prepos_liwc_prev', 'TotPron_liwc_prev', 'PronImp_liwc_prev',
                              'Adverb_liwc_prev', 'Negacio_liwc_prev', 'Incl_liwc_prev', 'Relativ_liwc_prev',
                              'Social_liwc_prev', 'MecCog_liwc_prev', 'Afect_liwc_prev', 'EmoPos_liwc_prev',
                              'Conjunc_liwc_prev', 'Excl_liwc_prev', 'Discrep_liwc_prev', 'Tiempo_liwc_prev',
                              'Causa_liwc_prev', 'Tentat_liwc_prev', 'Insight_liwc_prev', 'Certeza_liwc_prev',
                              'Biolog_liwc_prev', 'VerbAux_liwc_prev', 'Salud_liwc_prev', 'Sexual_liwc_prev',
                              'Cuerpo_liwc_prev', 'gender_demo_prev', 'age_demo_prev', 'retweets_count_social_prev',
                              'favs_count_social_prev', 'mentions_count_social_prev', 'ratio_quoted_tweet_types_prev',
                              'ratio_retweets_tweet_types_prev', 'ratio_replies_tweet_types_prev',
                              'ratio_original_tweet_types_prev', 'week_days_count_ratio_behav_prev',
                              'weekend_days_count_ratio_behav_prev', 'median_time_betweet_tweets_behav_prev',
                              'tweets_sleep_time_ratio_behav_prev', 'tweets_awake_time_ratio_behav_prev', 'lang_prev',
                              'legitimacy_prev', 'reputation_prev', 'status_prev', 'is_usual_suspect_curr',
                              'is_verified_curr', 'followers_count_curr', 'friends_count_curr', 'listed_count_curr',
                              'favorites_count_curr', 'statuses_count_curr', 'url_in_profile_curr', 'help_empath_curr',
                              'office_empath_curr', 'dance_empath_curr', 'money_empath_curr', 'wedding_empath_curr',
                              'domestic_work_empath_curr', 'sleep_empath_curr', 'medical_emergency_empath_curr',
                              'cold_empath_curr', 'hate_empath_curr', 'cheerfulness_empath_curr',
                              'aggression_empath_curr', 'occupation_empath_curr', 'envy_empath_curr',
                              'anticipation_empath_curr', 'family_empath_curr', 'vacation_empath_curr',
                              'crime_empath_curr', 'attractive_empath_curr', 'masculine_empath_curr',
                              'prison_empath_curr', 'health_empath_curr', 'pride_empath_curr', 'dispute_empath_curr',
                              'nervousness_empath_curr', 'government_empath_curr', 'weakness_empath_curr',
                              'horror_empath_curr', 'swearing_terms_empath_curr', 'leisure_empath_curr',
                              'suffering_empath_curr', 'royalty_empath_curr', 'wealthy_empath_curr',
                              'tourism_empath_curr', 'furniture_empath_curr', 'school_empath_curr', 'magic_empath_curr',
                              'beach_empath_curr', 'journalism_empath_curr', 'morning_empath_curr',
                              'banking_empath_curr', 'social_media_empath_curr', 'exercise_empath_curr',
                              'night_empath_curr', 'kill_empath_curr', 'blue_collar_job_empath_curr', 'art_empath_curr',
                              'ridicule_empath_curr', 'play_empath_curr', 'computer_empath_curr', 'college_empath_curr',
                              'optimism_empath_curr', 'stealing_empath_curr', 'real_estate_empath_curr',
                              'home_empath_curr', 'divine_empath_curr', 'sexual_empath_curr', 'fear_empath_curr',
                              'irritability_empath_curr', 'superhero_empath_curr', 'business_empath_curr',
                              'driving_empath_curr', 'pet_empath_curr', 'childish_empath_curr', 'cooking_empath_curr',
                              'exasperation_empath_curr', 'religion_empath_curr', 'hipster_empath_curr',
                              'internet_empath_curr', 'surprise_empath_curr', 'reading_empath_curr',
                              'worship_empath_curr', 'leader_empath_curr', 'independence_empath_curr',
                              'movement_empath_curr', 'body_empath_curr', 'noise_empath_curr', 'eating_empath_curr',
                              'medieval_empath_curr', 'zest_empath_curr', 'confusion_empath_curr', 'water_empath_curr',
                              'sports_empath_curr', 'death_empath_curr', 'healing_empath_curr', 'legend_empath_curr',
                              'heroic_empath_curr', 'celebration_empath_curr', 'restaurant_empath_curr',
                              'violence_empath_curr', 'programming_empath_curr', 'dominant_heirarchical_empath_curr',
                              'military_empath_curr', 'neglect_empath_curr', 'swimming_empath_curr',
                              'exotic_empath_curr', 'love_empath_curr', 'hiking_empath_curr',
                              'communication_empath_curr', 'hearing_empath_curr', 'order_empath_curr',
                              'sympathy_empath_curr', 'hygiene_empath_curr', 'weather_empath_curr',
                              'anonymity_empath_curr', 'trust_empath_curr', 'ancient_empath_curr',
                              'deception_empath_curr', 'fabric_empath_curr', 'air_travel_empath_curr',
                              'fight_empath_curr', 'dominant_personality_empath_curr', 'music_empath_curr',
                              'vehicle_empath_curr', 'politeness_empath_curr', 'toy_empath_curr', 'farming_empath_curr',
                              'meeting_empath_curr', 'war_empath_curr', 'speaking_empath_curr', 'listen_empath_curr',
                              'urban_empath_curr', 'shopping_empath_curr', 'disgust_empath_curr', 'fire_empath_curr',
                              'tool_empath_curr', 'phone_empath_curr', 'gain_empath_curr', 'sound_empath_curr',
                              'injury_empath_curr', 'sailing_empath_curr', 'rage_empath_curr', 'science_empath_curr',
                              'work_empath_curr', 'appearance_empath_curr', 'valuable_empath_curr',
                              'warmth_empath_curr', 'youth_empath_curr', 'sadness_empath_curr', 'fun_empath_curr',
                              'emotional_empath_curr', 'joy_empath_curr', 'affection_empath_curr',
                              'traveling_empath_curr', 'fashion_empath_curr', 'ugliness_empath_curr',
                              'lust_empath_curr', 'shame_empath_curr', 'torment_empath_curr', 'economics_empath_curr',
                              'anger_empath_curr', 'politics_empath_curr', 'ship_empath_curr', 'clothing_empath_curr',
                              'car_empath_curr', 'strength_empath_curr', 'technology_empath_curr',
                              'breaking_empath_curr', 'shape_and_size_empath_curr', 'power_empath_curr',
                              'white_collar_job_empath_curr', 'animal_empath_curr', 'party_empath_curr',
                              'terrorism_empath_curr', 'smell_empath_curr', 'disappointment_empath_curr',
                              'poor_empath_curr', 'plant_empath_curr', 'pain_empath_curr', 'beauty_empath_curr',
                              'timidity_empath_curr', 'philosophy_empath_curr', 'negotiate_empath_curr',
                              'negative_emotion_empath_curr', 'cleaning_empath_curr', 'messaging_empath_curr',
                              'competing_empath_curr', 'law_empath_curr', 'friends_empath_curr', 'payment_empath_curr',
                              'achievement_empath_curr', 'alcohol_empath_curr', 'liquid_empath_curr',
                              'feminine_empath_curr', 'weapon_empath_curr', 'children_empath_curr',
                              'monster_empath_curr', 'ocean_empath_curr', 'giving_empath_curr',
                              'contentment_empath_curr', 'writing_empath_curr', 'rural_empath_curr',
                              'positive_emotion_empath_curr', 'musical_empath_curr', 'surprise_emolex_curr',
                              'trust_emolex_curr', 'positive_emolex_curr', 'anger_emolex_curr', 'disgust_emolex_curr',
                              'anticipation_emolex_curr', 'sadness_emolex_curr', 'joy_emolex_curr',
                              'negative_emolex_curr', 'fear_emolex_curr', 'negative_sentiment_curr',
                              'neutral_sentiment_curr', 'positive_sentiment_curr', 'NOT-HATE_hate_sp_curr',
                              'hate_hate_sp_curr', 'Verbos_liwc_curr', 'Present_liwc_curr', 'verbosEL_liwc_curr',
                              'Funct_liwc_curr', 'Prepos_liwc_curr', 'TotPron_liwc_curr', 'PronImp_liwc_curr',
                              'Adverb_liwc_curr', 'Negacio_liwc_curr', 'Incl_liwc_curr', 'Relativ_liwc_curr',
                              'Social_liwc_curr', 'MecCog_liwc_curr', 'Afect_liwc_curr', 'EmoPos_liwc_curr',
                              'Conjunc_liwc_curr', 'Excl_liwc_curr', 'Discrep_liwc_curr', 'Tiempo_liwc_curr',
                              'Causa_liwc_curr', 'Tentat_liwc_curr', 'Insight_liwc_curr', 'Certeza_liwc_curr',
                              'Biolog_liwc_curr', 'VerbAux_liwc_curr', 'Salud_liwc_curr', 'Sexual_liwc_curr',
                              'Cuerpo_liwc_curr', 'gender_demo_curr', 'age_demo_curr', 'retweets_count_social_curr',
                              'favs_count_social_curr', 'mentions_count_social_curr', 'ratio_quoted_tweet_types_curr',
                              'ratio_retweets_tweet_types_curr', 'ratio_replies_tweet_types_curr',
                              'ratio_original_tweet_types_curr', 'week_days_count_ratio_behav_curr',
                              'weekend_days_count_ratio_behav_curr', 'median_time_betweet_tweets_behav_curr',
                              'tweets_sleep_time_ratio_behav_curr', 'tweets_awake_time_ratio_behav_curr', 'lang_curr',
                              'legitimacy_curr', 'reputation_curr', 'status_curr', 'is_usual_suspect_original',
                              'is_verified_original', 'followers_count_original', 'friends_count_original',
                              'listed_count_original', 'favorites_count_original', 'statuses_count_original',
                              'url_in_profile_original', 'help_empath_original', 'office_empath_original',
                              'dance_empath_original', 'money_empath_original', 'wedding_empath_original',
                              'domestic_work_empath_original', 'sleep_empath_original',
                              'medical_emergency_empath_original', 'cold_empath_original', 'hate_empath_original',
                              'cheerfulness_empath_original', 'aggression_empath_original',
                              'occupation_empath_original', 'envy_empath_original', 'anticipation_empath_original',
                              'family_empath_original', 'vacation_empath_original', 'crime_empath_original',
                              'attractive_empath_original', 'masculine_empath_original', 'prison_empath_original',
                              'health_empath_original', 'pride_empath_original', 'dispute_empath_original',
                              'nervousness_empath_original', 'government_empath_original', 'weakness_empath_original',
                              'horror_empath_original', 'swearing_terms_empath_original', 'leisure_empath_original',
                              'suffering_empath_original', 'royalty_empath_original', 'wealthy_empath_original',
                              'tourism_empath_original', 'furniture_empath_original', 'school_empath_original',
                              'magic_empath_original', 'beach_empath_original', 'journalism_empath_original',
                              'morning_empath_original', 'banking_empath_original', 'social_media_empath_original',
                              'exercise_empath_original', 'night_empath_original', 'kill_empath_original',
                              'blue_collar_job_empath_original', 'art_empath_original', 'ridicule_empath_original',
                              'play_empath_original', 'computer_empath_original', 'college_empath_original',
                              'optimism_empath_original', 'stealing_empath_original', 'real_estate_empath_original',
                              'home_empath_original', 'divine_empath_original', 'sexual_empath_original',
                              'fear_empath_original', 'irritability_empath_original', 'superhero_empath_original',
                              'business_empath_original', 'driving_empath_original', 'pet_empath_original',
                              'childish_empath_original', 'cooking_empath_original', 'exasperation_empath_original',
                              'religion_empath_original', 'hipster_empath_original', 'internet_empath_original',
                              'surprise_empath_original', 'reading_empath_original', 'worship_empath_original',
                              'leader_empath_original', 'independence_empath_original', 'movement_empath_original',
                              'body_empath_original', 'noise_empath_original', 'eating_empath_original',
                              'medieval_empath_original', 'zest_empath_original', 'confusion_empath_original',
                              'water_empath_original', 'sports_empath_original', 'death_empath_original',
                              'healing_empath_original', 'legend_empath_original', 'heroic_empath_original',
                              'celebration_empath_original', 'restaurant_empath_original', 'violence_empath_original',
                              'programming_empath_original', 'dominant_heirarchical_empath_original',
                              'military_empath_original', 'neglect_empath_original', 'swimming_empath_original',
                              'exotic_empath_original', 'love_empath_original', 'hiking_empath_original',
                              'communication_empath_original', 'hearing_empath_original', 'order_empath_original',
                              'sympathy_empath_original', 'hygiene_empath_original', 'weather_empath_original',
                              'anonymity_empath_original', 'trust_empath_original', 'ancient_empath_original',
                              'deception_empath_original', 'fabric_empath_original', 'air_travel_empath_original',
                              'fight_empath_original', 'dominant_personality_empath_original', 'music_empath_original',
                              'vehicle_empath_original', 'politeness_empath_original', 'toy_empath_original',
                              'farming_empath_original', 'meeting_empath_original', 'war_empath_original',
                              'speaking_empath_original', 'listen_empath_original', 'urban_empath_original',
                              'shopping_empath_original', 'disgust_empath_original', 'fire_empath_original',
                              'tool_empath_original', 'phone_empath_original', 'gain_empath_original',
                              'sound_empath_original', 'injury_empath_original', 'sailing_empath_original',
                              'rage_empath_original', 'science_empath_original', 'work_empath_original',
                              'appearance_empath_original', 'valuable_empath_original', 'warmth_empath_original',
                              'youth_empath_original', 'sadness_empath_original', 'fun_empath_original',
                              'emotional_empath_original', 'joy_empath_original', 'affection_empath_original',
                              'traveling_empath_original', 'fashion_empath_original', 'ugliness_empath_original',
                              'lust_empath_original', 'shame_empath_original', 'torment_empath_original',
                              'economics_empath_original', 'anger_empath_original', 'politics_empath_original',
                              'ship_empath_original', 'clothing_empath_original', 'car_empath_original',
                              'strength_empath_original', 'technology_empath_original', 'breaking_empath_original',
                              'shape_and_size_empath_original', 'power_empath_original',
                              'white_collar_job_empath_original', 'animal_empath_original', 'party_empath_original',
                              'terrorism_empath_original', 'smell_empath_original', 'disappointment_empath_original',
                              'poor_empath_original', 'plant_empath_original', 'pain_empath_original',
                              'beauty_empath_original', 'timidity_empath_original', 'philosophy_empath_original',
                              'negotiate_empath_original', 'negative_emotion_empath_original',
                              'cleaning_empath_original', 'messaging_empath_original', 'competing_empath_original',
                              'law_empath_original', 'friends_empath_original', 'payment_empath_original',
                              'achievement_empath_original', 'alcohol_empath_original', 'liquid_empath_original',
                              'feminine_empath_original', 'weapon_empath_original', 'children_empath_original',
                              'monster_empath_original', 'ocean_empath_original', 'giving_empath_original',
                              'contentment_empath_original', 'writing_empath_original', 'rural_empath_original',
                              'positive_emotion_empath_original', 'musical_empath_original', 'surprise_emolex_original',
                              'trust_emolex_original', 'positive_emolex_original', 'anger_emolex_original',
                              'disgust_emolex_original', 'anticipation_emolex_original', 'sadness_emolex_original',
                              'joy_emolex_original', 'negative_emolex_original', 'fear_emolex_original',
                              'negative_sentiment_original', 'neutral_sentiment_original',
                              'positive_sentiment_original', 'NOT-HATE_hate_sp_original', 'hate_hate_sp_original',
                              'Verbos_liwc_original', 'Present_liwc_original', 'verbosEL_liwc_original',
                              'Funct_liwc_original', 'Prepos_liwc_original', 'TotPron_liwc_original',
                              'PronImp_liwc_original', 'Adverb_liwc_original', 'Negacio_liwc_original',
                              'Incl_liwc_original', 'Relativ_liwc_original', 'Social_liwc_original',
                              'MecCog_liwc_original', 'Afect_liwc_original', 'EmoPos_liwc_original',
                              'Conjunc_liwc_original', 'Excl_liwc_original', 'Discrep_liwc_original',
                              'Tiempo_liwc_original', 'Causa_liwc_original', 'Tentat_liwc_original',
                              'Insight_liwc_original', 'Certeza_liwc_original', 'Biolog_liwc_original',
                              'VerbAux_liwc_original', 'Salud_liwc_original', 'Sexual_liwc_original',
                              'Cuerpo_liwc_original', 'gender_demo_original', 'age_demo_original',
                              'retweets_count_social_original', 'favs_count_social_original',
                              'mentions_count_social_original', 'ratio_quoted_tweet_types_original',
                              'ratio_retweets_tweet_types_original', 'ratio_replies_tweet_types_original',
                              'ratio_original_tweet_types_original', 'week_days_count_ratio_behav_original',
                              'weekend_days_count_ratio_behav_original', 'median_time_betweet_tweets_behav_original',
                              'tweets_sleep_time_ratio_behav_original', 'tweets_awake_time_ratio_behav_original',
                              'lang_original', 'legitimacy_original', 'reputation_original', 'status_original',
                              ]

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

        logger.info(f'Found {len(positive_samples)} positive samples')

        logger.info('Generating negative samples')
        negative_rows = self._generate_negative_rows(positive_samples)
        negative_users = list(set(negative_rows['source'].unique()) | set(negative_rows['target'].unique()))
        logger.info(f'Fetching user features for {len(negative_users)} negative users')
        negative_user_features = self.fetch_user_features(negative_users)
        logger.info(f'Fetched {len(negative_user_features)} negative user features')

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

        logger.info(f'Found {len(negative_samples)} negative samples')

        positive_samples['propagated'] = 1
        negative_samples['propagated'] = 0
        dataset = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        dataset = dataset.drop(columns=['source', 'target', 'cascade_id', 'original_author'], errors='ignore')
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        logger.info(f'Generated dataset with {len(dataset)} samples')
        return dataset[self.feature_names + ['propagated']]

    def get_rows(self, cascades):
        logger.info(f'Fetching rows for {len(cascades)} cascades')

        if self.num_samples is not None:
            total = self.num_samples
        else:
            total = len(cascades)
        rows = []
        for i, cascade in (pbar := tqdm(cascades.sample(frac=1).iterrows(), total=total)):
            samples = self._get_row_for_cascades(cascade)
            rows.extend(samples)
            if self.num_samples is not None:
                if len(rows) >= self.num_samples // 2:
                    break
                else:
                    pbar.update(len(samples))

            else:
                pbar.update(1)
        rows = pd.DataFrame(rows)
        logger.info(f'Found {len(rows)} rows')
        return rows

    def _generate_negative_rows(self, features):
        hidden_network = self.egonet.get_hidden_network(self.dataset)
        negatives = []
        sources = [(source, interactions) for source, interactions in features.groupby('source')]
        random.shuffle(sources)

        for source, interactions in tqdm(sources):
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
                if self.num_samples is not None and len(negatives) >= self.num_samples // 2:
                    break

        if not negatives:
            raise RuntimeError('No negative samples found')
        negatives = pd.DataFrame(negatives).astype(str)
        return negatives

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
        tweet_ids = cascades['tweet_id'].tolist()
        pipeline = [
            {'$match': {'id_str': {'$in': tweet_ids}}},
            {'$project': {'_id': 0, 'tweet_id': '$id_str'}}
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

    def _get_row_for_cascades(self, cascade):
        rows = []
        prop_tree = self._get_propagation_tree(cascade['tweet_id'])
        nodes = list(range(len(prop_tree.vs)))
        random.shuffle(nodes)
        for node in nodes:
            node = prop_tree.vs[node]
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

    def get_propagation_tree(self, tweet_id):
        return self.diffusion_metrics.get_propagation_tree(self.dataset, tweet_id)

    def get_neighbours(self, author_id):
        return self.egonet.get_neighbours(self.dataset, author_id)

    def get_features_for(self, tweet_id, op, sources, targets):
        tweet_features = self.fetch_tweet_features([tweet_id])
        user_features = self.fetch_user_features(sources + targets + [op])
        sources = pd.Series(sources, name='source')
        targets = pd.Series(targets, name='target')
        features = pd.concat([sources, targets], axis=1)
        features['cascade_id'] = tweet_id
        features['op'] = op
        features = features[['cascade_id', 'op', 'source', 'target']]
        features = features.merge(tweet_features, left_on='cascade_id', right_index=True, how='inner')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_prev'),
                                  left_on='source',
                                  right_index=True, how='inner')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_curr'),
                                  left_on='target',
                                  right_index=True, how='inner')
        features = features.merge(user_features.rename(columns=lambda x: f'{x}_original'),
                                  left_on='op',
                                  right_index=True, how='inner')
        features = features.drop(columns=['source', 'target', 'cascade_id', 'op'], errors='ignore')
        features = features.reindex(self.feature_names, axis=1)
        return features


class PropagationModel:
    def __init__(self, host='localhost', port=27017, reference_types=('replied_to', 'quoted', 'retweeted'),
                 use_profiling='full', use_textual='full'):
        self.host = host
        self.port = port
        self.reference_types = reference_types
        self.use_profiling = use_profiling
        self.use_textual = use_textual
        self.model = None

    def fit(self, X, y):
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('transformer', ColumnTransformer([
                ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                 X.select_dtypes(include='object').columns),
            ], remainder='passthrough', verbose_feature_names_out=False)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(scale_pos_weight=(len(y) - y.sum()) / y.sum(),
                                         n_estimators=4))
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


class CascadeGenerator:
    def __init__(self, model, dataset_generator):
        self.model = model
        self.dataset_generator = dataset_generator

    def generate_cascade(self, tweet_id):
        # Get the cascade of a given tweet
        prop_tree = self.dataset_generator.get_propagation_tree(tweet_id)
        prop_tree.vs['ground_truth'] = True
        visited_nodes = set(prop_tree.vs['author_id'])
        op = prop_tree.vs.find(tweet_id=tweet_id)['author_id']
        self._process_neighbour_propagation(tweet_id, op, op, prop_tree, visited_nodes)
        return prop_tree

    def _process_neighbour_propagation(self, tweet_id, op, author_id, cascade, visited_nodes):
        neighbours = self.dataset_generator.get_neighbours(author_id)
        available_neighbours = set(neighbours.keys()) - visited_nodes
        sources = [author_id] * len(available_neighbours)
        targets = list(available_neighbours)
        features = self.dataset_generator.get_features_for(tweet_id, op, sources, targets)
        if not features.empty:
            predictions = self.model.predict(features)
            predictions = pd.Series(predictions, index=targets)
            author_index = cascade.vs.find(author_id=author_id).index
            propagated = predictions[predictions == 1]
            visited_nodes.update(propagated.index.to_list())
            for target, prediction in propagated.items():
                if target not in cascade.vs['author_id']:
                    cascade.add_vertex(author_id=target, ground_truth=False)
                target_index = cascade.vs.find(author_id=target).index
                cascade.add_edge(author_index, target_index)
                visited_nodes.add(target)
                self._process_neighbour_propagation(tweet_id, op, target, cascade, visited_nodes)
        return cascade
