import os
import zipfile
from pathlib import Path
import pandas as pd
from twarc_csv import DataFrameConverter
from twarc_csv import CSVConverter


def flatten_tweets(data_jsonl):
    """Flatten the tweets from a zip JSONL file."""
    data_jsonl = Path(data_jsonl)
    from twarc_csv import CSVConverter

    with open(data_jsonl, "r") as infile:
        with open(data_jsonl.with_suffix('.csv'), "w") as outfile:
            converter = CSVConverter(infile=infile, outfile=outfile)
            converter.process()

