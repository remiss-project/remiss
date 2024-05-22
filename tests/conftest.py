import json
from datetime import datetime

from pymongo import MongoClient

from preprocess import unfix_timestamps


def populate_test_database(database_name):
    client = MongoClient('localhost', 27017)
    client.drop_database(database_name)
    db = client[database_name]
    collection = db.get_collection('raw')
    with open('test_resources/Openarms.mongoimport.jsonl') as f:
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
