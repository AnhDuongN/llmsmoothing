import logging
import csv
import argparse
import smooth
import tqdm
import torch
import prompt
import numpy as np
import certify
from question import Question

def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", help="probability for a word not to be substituted in smoothing distribution",type=float, default=0.5)
    parser.add_argument("-a_1", "--alpha_1", help="alpha_1, s.t. with probability 1-alpha_1, d(f(x) - f(bar(x)) < 2R) for all x-bar(x) leq r",
                        type = float, default = 0.05)
    parser.add_argument("-a_2", "--alpha_2", help="alpha_2 s.t. with probability 1-alpha_2, P(f(phi(X)) in MEB) > p",type=float, default=0.05)
    parser.add_argument("-D", "--delta", help="delta such that the fall of center hat(f) encloses 1/2 + delta probability \
                        mass of the smoothed f(x)",type=float, default = 0.01)
    
    parser.add_argument("-n", "--num_lines", help="number of original questions to be taken from dataset",type=int, default = 1)
    parser.add_argument("-N", "--smoothing_number", help="number of smoothed inputs to take",type=int, default=100)
    parser.add_argument("-m", "--quantile", help="number of samples to take q-th quantile to take to estimate the enclosing ball with probability 1- alpha_2 : see equation 8", \
                        type = int, default = 110) 
    
    parser.add_argument("-k", "--top_k", help="number of synonyms to be considered for smoothing",type=int, default=3)
    parser.add_argument("-r", "--radius", help="number of perturbations",type=int, default = 3)
    parser.add_argument("-c", "--search_exponent", help="search exponent for binary search",type=int, default = 30)

    parser.add_argument("-i", "--import_models", help="do import models", action="store_true")

    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser

if __name__ == "__main__":
    """
    Samples Z = {z_i} such that z_i follows a distribution f(\phi (x)) three times, twice for "smooth" step and once for "certify" step.
    For each question, outputs 3 files corresponding to the three Z samples named as question<question_number>_<1/2/3>.
    Computation of MEB and of the center of the ball is done in smooth.py, using question<question_number>_1 and question<question_number>_2.
    Computation of the max radius is done in certify.py, using question<question_number>_3.
    """
    parser = create_arg_parse()
    args = parser.parse_args()
    
    logger = logging.getLogger("__main__")

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    if args.import_models:
        from common import *
    if args.num_lines : 
        num_lines = args.num_lines -1
    else:
        num_lines = len(dataset)

    delta_1 = 2*np.exp(-2*args.smoothing_number*(args.alpha_1**2)) #see theorem 3

    ### Prompt
    logger.debug("Reached generation loop")
    with open("output.csv", "w") as file_certify : 
        writer = csv.writer(file_certify)
        writer.writerow(["question", "center_answer", "ball_radius", "certified_radius", "probability"])
        file_certify.close()

    for i, row in enumerate(dataset):
        if (i >num_lines):
            break
        logger.debug(f"Current question : {row['question']}")
        frag_filename = "question"+str(i)

        current_question = Question(row['question'], row['answer']['normalized_aliases'], row['question_id'])
        current_question.generate_synonyms_albert(args.top_k)

        # first N sample for smooth algorithm
        first_sample_name = frag_filename + "_1"
        prompt.sample(current_question, args.smoothing_number, args.top_k, args.alpha, first_sample_name)  
        # second N sample for smooth algorithm
        second_sample_name = frag_filename + "_2"
        prompt.sample(current_question, args.smoothing_number, args.top_k, args.alpha, second_sample_name)  
        # first m sample for certify algorithm
        third_sample_name = frag_filename + "_3"
        prompt.sample(current_question, args.smoothing_number, args.top_k, args.alpha, third_sample_name)  
    
        logger.debug("Reached smooth and certification phase")

        with open("output.csv", "a") as file_certify : 
            writer = csv.writer(file_certify)
            center = smooth.smooth(args.delta, delta_1, first_sample_name, second_sample_name)
            radius = certify.fin_certify("question"+str(i)+"_3", center, args.radius, len(row['question'].split()), args.top_k, 
            args.alpha, int(args.delta * 100), args.search_exponent, args.alpha_2, args.quantile)
            writer.writerow([row['question'], center, args.radius, radius, 1-(args.alpha_1 + args.alpha_2)])
        file_certify.close()

