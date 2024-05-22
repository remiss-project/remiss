import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pymongoarrow.monkey
from tqdm import tqdm
from twarc import ensure_flattened

pymongoarrow.monkey.patch_all()


def load_metadata(metadata_file):
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
    return usual_suspects, parties


def retrieve_remiss_metadata(tweet, usual_suspects, parties):
    username = tweet['author']['username']
    remiss_metadata = {'is_usual_suspect': username in usual_suspects.index}
    if username in parties.index:
        party = parties.loc[username, 'PARTIT']
        # check we have only a party
        if isinstance(party, str):
            party = party.strip()
        elif isinstance(party, pd.Series):
            party = party.iloc[0]
        else:
            print(f'Unexpected type for party {party} for user {username}')
            party = 'Desconocido'
        remiss_metadata['party'] = party
    else:
        remiss_metadata['party'] = None
    return remiss_metadata


def _preprocess_line(line, outfile, media_outfile, mongoimport_outfile, output_usual_suspects_and_politicians,
                     usual_suspects, parties):
    line = json.loads(line)
    tweets = ensure_flattened(line)
    for tweet in tweets:
        remiss_metadata = retrieve_remiss_metadata(tweet, usual_suspects, parties)
        tweet['author']['remiss_metadata'] = remiss_metadata
        # store separately usual suspects and politicians
        if remiss_metadata['is_usual_suspect'] or remiss_metadata['party']:
            output_usual_suspects_and_politicians.write(json.dumps(tweet) + "\n")

        # store separately the media in another file
        if 'attachments' in tweet and 'media' in tweet['attachments']:
            media = {'id': tweet['id'],
                     'text': tweet['text'],
                     'author': tweet['author'],
                     'media': tweet['attachments']['media']}
            media_outfile.write(json.dumps(media) + "\n")
        try:
            outfile.write(json.dumps(tweet) + "\n")
            # fix timestamps for mongoimport
            fix_timestamps(tweet)
            mongoimport_outfile.write(json.dumps(tweet) + '\n')
        except TypeError as ex:
            print(f'Error processing tweet {tweet["id"]}: {ex}')
            print(tweet)


def preprocess_tweets(twitter_jsonl_zip, metadata_file):
    """
    Preprocess the tweets gathered by running twarc and store them as a single tweet per line in a jsonl file named
    <twitter_jsonl_zip>.preprocessed.jsonl.
    Also store the media related information in a separate file named <twitter_jsonl_zip>.media.jsonl.
    Finally, store the tweets in a format in which the dates are correctly imported as such can by mongoimport
    in a file named <twitter_jsonl_zip>.mongoimport.jsonl.
    Optionally, if metadata_file is provided, add the metadata containing whether a tweet comes from
    a usual suspect or a politician to the tweets.

    $ mongoimport --db test_remiss --collection test_tweets --file test.mongoimport.jsonl


    :param twitter_jsonl_zip: Source tweets in jsonl format, compressed in a zip file,as outputted by twarc
    :param metadata_file: Additional Laia metadata in xlsx format
    :return:
    """
    twitter_jsonl_zip = Path(twitter_jsonl_zip)
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            num_lines = sum(1 for _ in infile)
    usual_suspects, parties = load_metadata(metadata_file)

    output_jsonl = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.preprocessed.jsonl')
    output_media_urls = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.media.jsonl')
    output_mongoimport = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.mongoimport.jsonl')
    output_usual_suspects_and_politicians = twitter_jsonl_zip.parent / (
            twitter_jsonl_zip.stem.split('.')[0] + '.usual_suspects_and_politicians.jsonl')
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            with open(output_jsonl, "w") as outfile:
                with open(output_media_urls, "w") as media_outfile:
                    with open(output_mongoimport, "w") as mongoimport_outfile:
                        with open(output_usual_suspects_and_politicians, "w") as usual_suspects_and_politicians_outfile:
                            for line in tqdm(infile, total=num_lines):
                                _preprocess_line(line, outfile, media_outfile,
                                                 mongoimport_outfile, usual_suspects_and_politicians_outfile,
                                                 usual_suspects, parties)


def fix_timestamps(tweet):
    date_fields = {'created_at', 'editable_until', 'retrieved_at'}
    for field, value in tweet.items():
        if field in date_fields:
            if isinstance(value, dict):
                if '$date' not in value:
                    raise ValueError(f'Unexpected format in timestamp field {field}: {value}')
            else:
                date = value.replace(' ', 'T')
                # if not date.endswith('Z'):
                #     date += 'Z'
                tweet[field] = {'$date': date}
        elif isinstance(value, dict):
            fix_timestamps(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    fix_timestamps(item)


def unfix_timestamps(tweet):
    date_fields = {'created_at', 'editable_until', 'retrieved_at'}
    for field, value in tweet.items():
        if field in date_fields:
            if isinstance(value, dict):
                if '$date' not in value:
                    raise ValueError(f'Unexpected format in timestamp field {field}: {value}')
                else:
                    # Cast to datetime
                    tweet[field] = datetime.fromisoformat(value['$date'])

        elif isinstance(value, dict):
            unfix_timestamps(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    unfix_timestamps(item)


def generate_test_data(twitter_jsonl_zip, metadata_file, output_file=None, freq='1D', quantity=10):
    """
    Sample tweets from the twitter_jsonl_zip file according to the unit, bin and quantity parameters. Store
    them in a file to be imported into mongo using mongoimport for testing purposes.
    :param twitter_jsonl_zip:
    :param metadata_file:
    :param unit:
    :param bin:
    :param quantity:
    :return:
    """
    twitter_jsonl_zip = Path(twitter_jsonl_zip)
    usual_suspects, parties = load_metadata(metadata_file)
    if output_file:
        output_mongoimport = Path(output_file)
    else:
        output_mongoimport = twitter_jsonl_zip.parent / (
                    twitter_jsonl_zip.stem.split('.')[0] + '.mongoimport.test.jsonl')
    data = []
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            for line in tqdm(infile, desc='Sampling tweets'):
                tweets = ensure_flattened(json.loads(line))
                for tweet in tweets:
                    remiss_metadata = retrieve_remiss_metadata(tweet, usual_suspects, parties)
                    tweet['author']['remiss_metadata'] = remiss_metadata
                    created_at = datetime.fromisoformat(tweet['created_at'])
                    fix_timestamps(tweet)
                    data.append([created_at, tweet])

    df = pd.DataFrame(data, columns=['created_at', 'tweet'])
    sample = df.groupby(pd.Grouper(key='created_at', freq=freq)).apply(
        lambda x: x.sample(quantity)).reset_index(drop=True)
    sample = sample['tweet']
    with open(output_mongoimport, "w") as mongoimport_outfile:
        for tweet in sample:
            mongoimport_outfile.write(json.dumps(tweet) + '\n')
