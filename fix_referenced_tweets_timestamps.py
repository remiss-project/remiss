import fire
from pandas import Timestamp
from pymongo import MongoClient
from tqdm import tqdm


def fix_referenced_tweets_timestamps(host, port, db_name):
    client = MongoClient(host, port)
    raw = client.get_database(db_name).get_collection('raw')

    # go through all tweets and fix the timestamp formate to datetime of the referenced tweets
    referenced_tweets = list(raw.find({'referenced_tweets': {'$exists': True}}))
    for tweet in tqdm(referenced_tweets):
        changed = False
        for referenced_tweet in tweet['referenced_tweets']:
            if 'created_at' in referenced_tweet:
                referenced_tweet['created_at'] = Timestamp(referenced_tweet['created_at'])
                changed = True
        if changed:
            raw.replace_one({'id': tweet['id']}, tweet)

if __name__ == '__main__':
    fire.Fire(fix_referenced_tweets_timestamps)