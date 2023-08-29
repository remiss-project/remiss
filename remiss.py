import json
import os
import zipfile
from pathlib import Path
import pandas as pd
from twarc import ensure_flattened

from twarc_csv import DataFrameConverter
from twarc_csv import CSVConverter
from itertools import chain


def flatten_tweets(data_jsonl):
    """Flatten the tweets from a JSONL file."""
    data_jsonl = Path(data_jsonl)
    if not data_jsonl.exists():
        raise FileNotFoundError(f"{data_jsonl} does not exist.")

    with open(data_jsonl, "r") as infile:
        with open(data_jsonl.with_suffix('.flattened.jsonl'), "w") as outfile:
            for line in infile:
                line = json.loads(line)
                tweets = ensure_flattened(line)
                for tweet in tweets:
                    outfile.write(json.dumps(tweet) + "\n")
