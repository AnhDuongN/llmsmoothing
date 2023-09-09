# from common import dataset
import pandas as pd
import csv
import unicodedata as ud
import regex as re
from datasets import load_dataset

if __name__ == "__main__":
    """
    Exact matching to see if the ball centers correspond to the original answer(s) 
    from the dataset.
    """

    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    list_answers = []
    with open("output_certify.csv", "r") as f:
        reader = csv.reader(f)
        for i, (row, answer) in enumerate(zip(reader, dataset)):
            if re.sub(r'[^A-Za-z0-9 ]+', '',row[1]).lower() in answer['normalized_aliases']:
                list_answers.append(True)
            else:
                list_answers.append(False)
    df = pd.read_csv("output_certify.csv")
    df['entailement'] = list_answers
    df.to_csv("entailment.csv")