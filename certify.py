import pandas as pd
import compute_rho
import numpy as np
import csv
import argparse
import logging
import json


def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--radius", help="number of perturbations",type=int, default = 3)
    parser.add_argument("-k", "--k", help="number of synonyms to be considered for smoothing", type = int, default = 10) 
    parser.add_argument("-a", "--alpha", help="probability for a word not to be substituted in smoothing distribution",type=float, default=0.75)
    parser.add_argument("-D", "--delta_times_100", help="delta such that the fall of center hat(f) encloses 1/2 + delta probability \
                        mass of the smoothed f(x) TIMES 100", type=int, default=5)
    parser.add_argument("-c", "--search_exponent", help="search exponent for binary search",type=int, default = 30)
    parser.add_argument("-A", "--alpha_2", help="alpha_2 s.t. with probability 1-alpha_2, P(f(phi(X)) in MEB) > p",type=float, default=0.05)
    parser.add_argument("-m", "--quantile", help="number of samples to take q-th quartile to take to estimate the enclosing ball with probability 1- alpha_2 : see equation 8", \
                        type = int, default = 100) 
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser

def certify(filename : str, center : str) -> list[float]:
    """
    Computes the distances betweeb the center and each answer in <filename>
    Parameters : 
    - filename : dataset containing the answers from the sampling for certify algorithm (Algorithm 3)
    - center   : center of the MEB computed in smooth.py
    Returns : List of distances from center to each answer
    """
    data = pd.read_csv(filename)
    answers = data['answer'].tolist()
    distances = []
    for _, answer in enumerate(answers):
        dist = compute_wmd(center, answer)
        if dist != float('inf'):
            distances.append(dist)
        else:
            logger.debug("Inf distance!")
    return distances

def fin_certify(filename : str, center : str,  r : int, d : int, k : int, alpha : float, 
                delta_times_100 : int, search_exponent : int, alpha_2 : float, m : int) -> float:
    """
    Implementation of algorithm 3 in article. Returns the radius of the ball centered on center 
    such that if the perturbation is bounded by r, the output ball has a radius bounded by the 
    return value of this function.

    Parameters : 
    - filename        : dataset of sample for certify step
    - center          : center of the MEB computed in algorithm 2 (smooth.py)
    - r               : number of words that are perturbed
    - d               : length of input question
    - k               : Number of synonyms considered at substitution in smoothing step
    - alpha           : probability that a word from the canonical sentence isn't substituted (same alpha as in Question.smoothN())
    - delta_times_100 : delta such that the fall of center hat(f) encloses 1/2 + delta probability mass of the smoothed f(x) TIMES 100
    - search_exponent : search exponent for binary search
    - alpha_2         : alpha_2 s.t. with probability 1-alpha_2, P(f(phi(X)) in MEB) > p
    - m               : number of samples to take q-th quartile to take to estimate the enclosing ball with probability 1- alpha_2 : see equation 8
    """

    beta = (1-alpha)/k

    normalized_p = compute_rho.compute_rho_normalized(r, d, k, alpha, delta_times_100)
    p = compute_rho.return_to_base(normalized_p, search_exponent, k, d)

    radii = certify(filename, center)

    q = p + np.sqrt(np.log(1/alpha_2)/(2*m))
    logger.debug(f"p : {p}, q : {q}")
    logger.debug(f"radii : {radii}")
    return (1+beta)*np.quantile(radii, q)

if __name__ == "__main__":
    parser = create_arg_parse()
    args = parser.parse_args()
    logger = logging.getLogger("__certify__")

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    if args.import_models:
        from common import compute_wmd

    with open('config_prompt.json', 'r') as f:
        config = json.load(f)
        assert(config["alpha"] == args.alpha), "prompt.py was executed with a different value for alpha"
        assert(config["m"] == args.quantile), "prompt.py was executed with a different value for m"
        assert(config["k"] == args.k), "prompt.py was executed with a different value for k"
        f.close()
    
    with open('config_smooth.json', 'r') as f:
        config = json.load(f)
        assert(int(eval(config["delta"])*100) == args.delta_times_100), "smooth.py was executed with a different value for delta"
        f.close()

    with open("output_certify.csv", "w") as f:
        writer = csv.writer(f)
        with open("smooth.csv", "r") as g:
            reader_smooth = csv.reader(g)
            row1 = next(reader_smooth)
            for i, row in enumerate(reader_smooth):
                question, center= row[0], row[1]
                radius = fin_certify("question"+str(i)+"_3", center, args.radius, len(question.split()), args.k, 
                            args.alpha, args.delta_times_100, args.search_exponent, args.alpha_2, args.quartile)
                writer.writerow([question, center, radius])
            g.close()
        f.close()
