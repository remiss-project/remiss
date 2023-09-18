import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from twarc import ensure_flattened
import zipfile


def preprocess_tweets(twitter_jsonl_zip, metadata_file=None):
    """Flatten the tweets from a JSONL zipped file."""
    twitter_jsonl_zip = Path(twitter_jsonl_zip)
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            num_lines = sum(1 for _ in infile)
    if metadata_file:
        raw_metadata = pd.read_excel(metadata_file, sheet_name=None)
        # usual_suspects = []
        # for sheet_name, df in raw_metadata.items():
        #     if 'USUAL SUSPECTS' in sheet_name:
        #         usual_suspects.append(df)
        # usual_suspects = pd.concat(usual_suspects)
        usual_suspects = raw_metadata['NOVA LLISTA USUAL SUSPECTS']
        usual_suspects = usual_suspects.dropna(axis=1, how='all')
        usual_suspects = usual_suspects[usual_suspects['XARXA SOCIAL'] == 'Twitter']
        usernames = usual_suspects['ENLLAÇ'].str.split('/').str[-1].str.split('?').str[0]
        usual_suspects = usual_suspects.set_index(usernames)

        if not usual_suspects.index.is_unique:
            print('WARNING: usual suspects usernames are not unique')
            print(usual_suspects[usual_suspects.index.duplicated(keep=False)].sort_index())
            usual_suspects = usual_suspects.drop_duplicates(keep='first')

        parties = raw_metadata['LLISTA POLÍTICS']
        parties = parties.dropna(axis=1, how='all')
        parties = parties[parties['ENLLAÇ TW'] != 'No en té']
        usernames = parties['ENLLAÇ TW'].str.split('/').str[-1].str.split('?').str[0]
        parties = parties.set_index(usernames)
        if not parties.index.is_unique:
            print('WARNING: parties usernames are not unique')
            print(parties[parties.index.duplicated(keep=False)].sort_index())
            parties = parties.drop_duplicates(keep='first')

    output_jsonl = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.preprocessed.jsonl')
    output_media_urls = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.media.jsonl')
    output_mongodbimport = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.mongodbimport.jsonl')
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            with open(output_jsonl, "w") as outfile:
                with open(output_media_urls, "w") as media_outfile:
                    with open(output_mongodbimport, "w") as mongodbimport_outfile:
                        for line in tqdm(infile, total=num_lines):
                            line = json.loads(line)
                            tweets = ensure_flattened(line)
                            for tweet in tweets:
                                # store separately the media in another file
                                if 'attachments' in tweet and 'media' in tweet['attachments']:
                                    media = {'id': tweet['id'],
                                             'media': tweet['attachments']['media']}
                                    media_outfile.write(json.dumps(media) + "\n")
                                if metadata_file:
                                    username = tweet['author']['username']
                                    remiss_metadata = {'is_usual_suspect': username in usual_suspects.index}
                                    if username in parties.index:
                                        remiss_metadata['party'] = parties.loc[username, 'PARTIT']
                                    else:
                                        remiss_metadata['party'] = None
                                    tweet['author']['remiss_metadata'] = remiss_metadata

                                outfile.write(json.dumps(tweet) + "\n")

                                # convert the timestamps to mongodbimport format
                                # "starttime": "2019-12-01 00:00:05.5640"
                                # to
                                # "starttime": {
                                #     "$date": "2019-12-01T00:00:05.5640Z"
                                # }
                                fix_timestamps(tweet)
                                mongodbimport_outfile.write(json.dumps(tweet) + '\n')


def fix_timestamps(tweet):
    date_fields = {'created_at', 'editable_until', 'retrieved_at'}
    for field, value in tweet.items():
        if field in date_fields:
            if isinstance(value, dict):
                if '$date' not in value:
                    raise ValueError(f'Unexpected format in timestamp field {field}: {value}')
            else:
                tweet[field] = {'$date': value.replace(' ', 'T') + 'Z'}
        elif isinstance(value, dict):
            fix_timestamps(value)