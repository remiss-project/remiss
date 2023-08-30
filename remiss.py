import json
from pathlib import Path
from tqdm import tqdm
from twarc import ensure_flattened
import zipfile


def flatten_tweets(twitter_jsonl_zip):
    """Flatten the tweets from a JSONL zipped file."""
    twitter_jsonl_zip = Path(twitter_jsonl_zip)
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            num_lines = sum(1 for _ in infile)

    output_jsonl = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.flattened.jsonl')
    output_media_urls = twitter_jsonl_zip.parent / (twitter_jsonl_zip.stem.split('.')[0] + '.media.jsonl')
    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            with open(output_jsonl, "w") as outfile:
                with open(output_media_urls, "w") as media_outfile:
                    for line in tqdm(infile, total=num_lines):
                        line = json.loads(line)
                        tweets = ensure_flattened(line)
                        for tweet in tweets:
                            # store separately the media in another file
                            if 'attachments' in tweet and 'media' in tweet['attachments']:
                                media = {'id': tweet['id'],
                                         'media': tweet['attachments']['media']}
                                media_outfile.write(json.dumps(media) + "\n")

                            outfile.write(json.dumps(tweet) + "\n")
