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

    with zipfile.ZipFile(twitter_jsonl_zip, 'r') as zip_ref:
        with zip_ref.open(zip_ref.namelist()[0], 'r') as infile:
            with open(twitter_jsonl_zip.with_name(twitter_jsonl_zip.stem.split('.')[0]  + '.flattened.jsonl'), "w") as outfile:
                for line in tqdm(infile, total=num_lines):
                    line = json.loads(line)
                    tweets = ensure_flattened(line)
                    for tweet in tweets:
                        outfile.write(json.dumps(tweet) + "\n")
