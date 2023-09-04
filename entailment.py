import csv
from common import dataset


if __name__ == "__main__":
    list_answers = []
    with open("output.csv", "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            norm = dataset[i]
            inferred_answer = line[2]
            if (inferred_answer in norm):
                list_answers.append(True)

    for i, row in enumerate(dataset):
        normalised_answers = row['answer']['normalized_aliases']