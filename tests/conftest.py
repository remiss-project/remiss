import json
import shutil
import subprocess
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

from preprocess import unfix_timestamps


def create_test_database(source_dataset, database_name):
    # Connect to prod mongodb at mongodb://srvinv02.esade.es:27017/
    prod = MongoClient(host='srvinv02.esade.es', port=27017)
    prod_db = prod[source_dataset]
    # for each collection, sample the first week and insert into the test database at the local database_name
    client = MongoClient('localhost', 27017)
    client.drop_database(database_name)
    db = client[database_name]
    raw = prod_db.get_collection('raw')
    first_raw_tweet = raw.find_one(sort=[('created_at', 1)])
    last_raw_tweet = raw.find_one(sort=[('created_at', -1)])

    textual = prod_db.get_collection('textual')
    textual_dates = textual.aggregate_pandas_all([
        {'$project': {'date': 1}},
    ])
    textual_dates = pd.to_datetime(textual_dates['date'])
    first_textual_tweet = textual_dates.min()
    last_textual_tweet = textual_dates.max()

    first_tweet = max(first_raw_tweet['created_at'], first_textual_tweet)
    last_tweet = min(last_raw_tweet['created_at'], last_textual_tweet)

    middle_tweet = first_tweet + (last_tweet - first_tweet) / 2
    middle_plus_week = middle_tweet + pd.Timedelta(days=7)

    # Add all profiling data with their associated tweets, plus a week from the middle
    profiling = prod_db.get_collection('profiling')
    profiling_data = list(profiling.find())
    db.get_collection('profiling').insert_many(profiling_data)

    profiling_user_ids = [profiling['user_id'] for profiling in profiling_data]
    profiling_ids = raw.find({'author.id': {'$in': profiling_user_ids}}).distinct('id')
    raw_ids = raw.find({'created_at': {'$gte': first_tweet, '$lt': middle_plus_week}}).distinct('id')
    tweet_ids = profiling_ids + raw_ids
    raw_data = list(raw.find({'id': {'$in': tweet_ids}}))
    db.get_collection('raw').insert_many(raw_data)

    textual_data = textual.find({'id': {'$in': [int(tweet_id) for tweet_id in tweet_ids]}})
    db.get_collection('textual').insert_many(textual_data)

    # Add all multimodal data
    multimodal = prod_db.get_collection('multimodal')
    multimodal_data = list(multimodal.find())
    if len(multimodal_data) > 0:
        db.get_collection('multimodal').insert_many(multimodal_data)
    else:
        print('No multimodal data found')

    print('Done!')


def populate_test_database(database_name, small=False):
    client = MongoClient('localhost', 27017)
    client.drop_database(database_name)
    db = client[database_name]
    collection = db.get_collection('raw')
    path = 'test_resources/Openarms.mongoimport.jsonl' if not small else 'test_resources/Openarms.mongoimport.subset.jsonl'
    with open(path) as f:
        test_data = []
        for line in f:
            tweet = json.loads(line)
            unfix_timestamps(tweet)
            test_data.append(tweet)

    collection.insert_many(test_data)

    # # Add profiling data
    # collection = db.get_collection('profiling')
    # # import from profiling.json
    # with open('test_resources/profiling.json') as f:
    #     data = [json.loads(line) for line in f]

    collection = db.get_collection('profiling')
    with open('test_resources/profiling.json') as f:
        data = [json.loads(line) for line in f]

    collection.insert_many(data)

    collection = db.get_collection('textual')
    with open('test_resources/Openarms.textual.json') as f:
        data = json.load(f)
        for tweet in data:
            del tweet['_id']
            tweet['id'] = tweet['id']['$numberLong']

    collection.insert_many(data)

    # Add multimodal data
    collection = db.get_collection('multimodal')
    collection.insert_many([
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
        },
        {
            "claim_text": "A shot of the All Nippon Airways Boeing 787 Dreamliner that s painted in the likeness of R2D2 in Los Angeles on Dec 15 2015",
            "id": 67,
            "tweet_id": "100485426",
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
        },
    ])


def delete_test_database(database_name):
    client = MongoClient('localhost', 27017)
    client.drop_database(database_name)


# populate_test_database('test_dataset')

if __name__ == '__main__':
    create_test_database('Openarms', 'test_dataset_2')
