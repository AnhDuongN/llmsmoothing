# from common import dataset
import pandas as pd
import csv
import unicodedata as ud
import regex as re
from datasets import load_dataset
import logging
import argparse

if __name__ == "__main__":
    """
    Exact matching to see if the ball centers corresponds to the original answer(s) 
    from the dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0)
    logger = logging.getLogger("__entailment__")
    args = parser.parse_args()
    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    list_answers = []
    with open("output_certify.csv", "r") as f:
        reader = csv.reader(f)
        for i, (row, answer) in enumerate(zip(reader, dataset)):
            if i == 0:
                continue
            if re.sub(r'[^A-Za-z0-9 ]+', '',row[1]).lower() in answer['answer']['normalized_aliases']:
                list_answers.append(True)
            else:
                list_answers.append(False)
    df = pd.read_csv("output_certify.csv")
    logger.debug(df.head())
    logger.debug(list_answers)
    df['entailement'] = list_answers
    df.to_csv("entailment.csv")
