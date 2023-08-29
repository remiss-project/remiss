import json
from pathlib import Path
from tqdm import tqdm
from twarc import ensure_flattened



def flatten_tweets(data_jsonl):
    """Flatten the tweets from a JSONL file."""
    data_jsonl = Path(data_jsonl)
    if not data_jsonl.exists():
        raise FileNotFoundError(f"{data_jsonl} does not exist.")

    with open(data_jsonl, "r") as infile:
        num_lines = sum(1 for _ in infile)
    with open(data_jsonl, "r") as infile:
        with open(data_jsonl.with_suffix('.flattened.jsonl'), "w") as outfile:
            for line in tqdm(infile, total=num_lines):
                line = json.loads(line)
                tweets = ensure_flattened(line)
                for tweet in tweets:
                    outfile.write(json.dumps(tweet) + "\n")
