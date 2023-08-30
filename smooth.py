import pandas as pd
import argparse

import statistics
import logging
import numpy as np
import csv
from datasets import load_dataset



def smooth(delta : float, delta_1 : float, file_1 : str, file_2 : str) -> str:
    """
    Returns the computed f hat (center of enclosing ball) computed by algorithm 2.
    Parameters : 
    - delta   : delta as in paper
    - delta_1 : delta_1 as in paper (for probailistic estimation of the radius of the first ball)
    - file_1  : output file of the first N sample
    - file_2  : output file of the second N sample
    Returns : Center of the Minimum Enclosing Ball
    """
    center, radius = computeMEB(file_1)
    logger.debug(f"MEB : Center : {center}, Radius : {radius}")
    rho = compute_p(file_2, center, radius)
    logger.debug(f"p : {rho}")
    p_delta_1 = rho - delta_1
    delta_2 = 0.5 - p_delta_1
    logger.debug(f"delta_1 : {delta_1}, delta_2 : {delta_2}, delta : {delta}")
    if max(delta_1, delta_2) <= delta:
        return center
    else : 
        return "ABSTAINED FROM ANSWERING"

def computeMEB(filename : str) -> tuple[str, float]:
    """
    Computes the Minimum Enclosing Ball from the samples in the given file
    Parameters : 
    - filename : File with the samples from the first sampling
    Returns : Center of the minimum enclosing ball and estimated first radius of the minimum enclosing ball
    """
    logger.warning("Computing MEB : O(n^2) step")

    data = pd.read_csv(filename)
    answers = data['answer'].tolist() 

    # Allocating the space is slightly faster than appending
    medians = [None]*len(answers)   # Median distance from the answer to all other answers
    max_radii = [None]*len(answers) # Max distance from the answer to all other answers

    # Compute pair-wise distance for all answers to get the approximate center
    for i in range(len(answers)):
        temp_array = [None]*len(answers)
        for j in range(len(answers)):
            temp_array[j] = compute_wmd(answers[i], answers[j])
            logger.debug(f"Distance : {temp_array[j]}")
        medians[i] = statistics.median(temp_array)
        max_radii[i] = max(temp_array)

    # The approximated center is the point that has the minimum median distance from all the points in the set
    median_index = medians.index(min(medians)) 
    return answers[median_index], max_radii[median_index]

def compute_p(filename : str, center : str, radius : float) -> float:
    """
    Compute the proportion of points of the 2nd sampling that fall into the MEB of the 1st sampling
    Parameters : 
    - filename : results of the second sampling
    - center   : center of the MEB of the first sampling
    - radius   : radius of the MEB of the first sampling
    Returns : Proportion of points of the 2nd sampling that fall into the MEB of the 1st sampling
    """
    data = pd.read_csv(filename)
    answers = data['answer'].tolist()

    count = 0
    for i in range(len(answers)):
        if compute_wmd(answers[i], center) <= radius:
            count +=1
    return count/len(answers)

if __name__ == "__main__":
    """
    Smoothing step (Algorithm 2)
    Outputs a file, such that for each question, the "center" answer is given, and "ABSTAINED FROM ANSWERING" is given
    if smoothing is not possible.
    """
    logger = logging.getLogger("__smooth__")
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_lines", help="number of original questions to be taken from dataset",\
                        type=int, default = 10)
    parser.add_argument("-D", "--delta", help="delta such that the fall of center hat(f) encloses 1/2 + delta probability \
                        mass of the smoothed f(x)",type=float, default = 0.01)
    parser.add_argument("-a", "--alpha_1", help="alpha_1, s.t. with probability 1-alpha_1, d(f(x) - f(bar(x)) < 2R) for all x-bar(x) leq r",
                        type = float, default = 0.001)
    parser.add_argument("-N", "--N", help="number of smoothed inputs to take",type=int, default = 100)
    parser.add_argument("-i", "--insert_model", help="import models", action="store_true")
    args = parser.parse_args()

    delta_1 = 2*np.exp(-2*args.N*(args.alpha_1**2)) #see theorem 3

    if args.insert_model:
        from common import dataset, compute_wmd
    if args.num_lines : 
        num_lines = args.num_lines -1
    else:
        num_lines = len(dataset)

    logger.debug("Reached generation loop")
    
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
    
