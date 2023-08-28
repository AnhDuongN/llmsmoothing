import pandas as pd
import argparse

import statistics
import logging
import numpy as np
import csv
from datasets import load_dataset
import common



def smooth(delta : float, delta_1 : float, file_1 : str, file_2 : str):
    """
    """
    center, radius = computeMEB(file_1)
    rho = compute_p(file_2, center, radius)
    p_delta_1 = rho - delta_1
    delta_2 = 0.5 - p_delta_1
    if max(delta_1, delta_2) <= delta:
        return center
    else : 
        return "ABSTAINED FROM ANSWERING"

def computeMEB(filename : str):
    """
    Computes the MEB
    """
    data = pd.read_csv(filename)
    answers = data['answer'].tolist()
    medians = [None]*len(answers)
    for i in range(len(answers)):
        temp_array = [None]*len(answers)
        for j in range(len(answers)):
            temp_array[j] = common.compute_wmd(answers[i], answers[j])
        medians[i] = statistics.median(temp_array)
    median_radius = min(medians)
    median_index = medians.index(median_radius)
    return answers[median_index], median_radius

def compute_p(filename : str, center : str, radius : float) -> float:
    data = pd.read_csv(filename)
    answers = data['answer'].tolist()
    count = 0
    for i in range(len(answers)):
        if common.compute_wmd(answers[i], center) <= float:
            count +=1
    return count/len(answers)

if __name__ == "__main__":
    """
    Smoothing step of algorithm
    Outputs a file, such that for each question, the "center" answer is given, and "ABSTAINED FROM ANSWERING" is given
    if smoothing is not possible.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_lines", help="number of original questions to be taken from dataset, indexed from 0",type=int, default = 10)
    parser.add_argument("-D", "--delta", help="delta (in paper)",type=float, default = 0.01) #TODO better explanation
    parser.add_argument("-a", "--alpha_1", help="alpha_1, s.t. with probability 1-alpha_1, d(f(x) - f(bar(x)) < 2R) for all x-bar(x) leq r",
                        type = float, default = 0.001) #TODO better explanation
    parser.add_argument("-N", "--N", help="see N in prompt.py",type=int, default = 100) #TODO better explanation


    args = parser.parse_args()
    delta_1 = 2*np.exp(-2*args.N*(args.delta_1**2))

    ### Load questions and answers
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    if args.num_lines : 
        num_lines = args.num_lines
    else:
        num_lines = len(dataset)

    logging.debug("Reached generation loop")
    
    with open("smooth.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "center_answer"])
        for i, row in enumerate(dataset):
            if (i >num_lines):
                break
            frag_filename = "question"+str(i)
            first_sample_name = frag_filename + "_1"
            second_sample_name = frag_filename + "_2"
            writer.writerow([row['question'], smooth(args.delta, delta_1, first_sample_name, second_sample_name)])
        f.close()
    
