import math
import csv
import argparse
import logging


def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--radius", help="number of original questions to be taken from dataset, indexed from 0",type=int, default = 10)
    parser.add_argument("-d", "--dimension", help="delta (in paper)",type=int, default=14) #TODO better explanation
    parser.add_argument("-k", "--k", help="top_k", type = int, default = 10) 
    parser.add_argument("-a", "--alpha", help="alpha",type=float, default=0.75)
    parser.add_argument("-D", "--delta_times_100", help="100 x D", type=int, default=5)
    parser.add_argument("-c", "--search_exponent", help="search exponent for binary search", default = 30)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser

def compute_cardinality(u : int, v : int, r : int, d : int, k : int) -> int:
    """
    The cardinality of each region L(u;v;r). See paper (Lemma 11) for explanations. This is a direct port from lemma 11.
    Parameters : 
    - u : coordinates flipped from x_c (the canonical sentence)
    - v : coordinates flipped from bar(x_c) (the perturbed sentence)
    - r : perturbation radius
    - d : dimension (length of the vector)
    - k : top k synonyms chosen to smooth each word (see e.g. Question.smoothN or see args)
    Return : Cardinality of each region L(u;v;r)
    """
    cardinality = 0
    first_i = max(0, v-r)
    end_i = min(u, d-r, (u+v-r)//2)
    for i in range(first_i, end_i+1):
        j = u + v - 2*i - r 
        cardinality += ((k-1)**j * math.factorial(r) * k**i * math.factorial(d-r)) \
        / (math.factorial(u-i-j) * math.factorial(v-i-j) * math.factorial(j) * math.factorial(d-r-i) * math.factorial(i))
    return int(cardinality)

def compute_likelihood_ratio(u : int, v : int, k : int, alpha_tilda : float, beta_tilda : float) -> float:
    """
    Computes the likelihood ratio n(x, bar(x)) : For each z in X (input space) define 
    n(x, bar(x)) = P(phi(x) = z) / P(phi(bar(x)) = z)
    Parameters : 
    - u     : coordinates flipped from x_c (the canonical sentence)
    - v     : coordinates flipped from bar(x_c) (the perturbed sentence)
    - k     : top k synonyms chosen to smooth each word (see e.g. Question.smoothN or see args)
    - alpha : probability that a word from the canonical sentence isn't substituted (same alpha as in Question.smoothN())
    - beta  : probability of each substitute if the word in the canonical sentence is substituted (same beta as in Question.smoothN())
    Returns : 
    Likelihood ratio
    """
    # Numbers are multiplied by 100 *k to avoid floating point arithmetics errors
    return (alpha_tilda ** (v-u)) * (beta_tilda **(u-v))

def compute_cardinalities_and_ratios(r : int , d : int, k : int, alpha_tilda : float, beta_tilda : float) -> int:
    """
    Computes cardinalities and ratios for each tuple (u,v) in (0, d)^2.
    Parameters : 
    - r           : perturbation radius
    - d           : dimension (length of the vector)
    - k           : top k synonyms chosen to smooth each word (see e.g. Question.smoothN or see args)
    - alpha_tilda : probability that a word from the canonical sentence isn't substituted (same alpha as in Question.smoothN()) 
                    TIMES 100 TIMES K 
    - beta_tilda  : probability of each substitute if the word in the canonical sentence is substituted (same beta as in Question.smoothN()) 
                    TIMES 100 TIMES K
    Returns : 
    A file (temp_card.csv) such that for each line the values are : 
    [u, v, cardinality of L(u;v;r), likelihood ratio n(x, bar(x))]
    """
    with open("temp_card.csv", 'w') as f : #this file is only temporary
        writer = csv.writer(f)
        for u in range(0, d+1):
            for v in range(u, d+1):
                writer.writerow([u, v, compute_cardinality(u,v,r,d,k), compute_likelihood_ratio(u,v,k,alpha_tilda,beta_tilda)])
                if v!= u: #divides by 2 the number of operations needed
                    writer.writerow([v, u, compute_cardinality(u,v,r,d,k), compute_likelihood_ratio(v,u,k,alpha_tilda,beta_tilda)]) 
        f.close()

def compute_rho_normalized(r : int, d : int, k : int, alpha : float, beta : float, delta_times_100 : int) -> int:
    """
    Idk what this is (some kind of binary search) but this works better than following the original algorithm so whatever
    """
    # Normalization is done to avoid floating point arithmetics (if the probabilities are too small, floats aren't as accurate)
    alpha_tilda = alpha*100*k
    beta_tilda = beta*100*k
    half_Z = (50+delta_times_100)*k * (100*k)**(d-1)

    compute_cardinalities_and_ratios(r, d, k, alpha_tilda, beta_tilda)

    with open("temp_card.csv", "r") as f:
        reader = csv.reader(f)
        cardinalities = list(reader)
    f.close()

    cardinalities = sorted(cardinalities, key=lambda x : x[-1], reverse=True)
    # The following is based on the code of the GitHub of the MIT guys from the l0 paper (see threshold_mnist.py)
    p_given, q_worst = 0,0
    for _, cardinality in enumerate(cardinalities):
        cardinality = [eval(i) for i in cardinality]
        p = int(alpha_tilda**(d-cardinality[0]) * beta_tilda**(cardinality[0]))
        q = int(alpha_tilda**(d-cardinality[1]) * beta_tilda**(cardinality[1]))
        q_delta = q * cardinality[2]

        if q_delta < half_Z:
            half_Z -= q_delta
            p_given += p * cardinality[2]

        else:
            upper = cardinality[2]
            lower = 0
            while upper-lower != 1:
                mid = (lower+upper) >> 1
                running_Z = q_worst + mid*q
                if running_Z < half_Z:
                    lower = mid
                else:
                    upper = mid
            q_worst += upper*q
            p_given += upper*p
            return p_given
        
    logging.critical("No p was computed!")
    return -1

def return_to_base(p : int, c : int, k : int, d : int):
    """
    Binary search to return the computed threshold to the original interval [0,1)
    """
    upper, lower = 10 ** c, 0
    cst = ((10*k) ** c) * ((100*k)**(d-c))
    while upper-lower != 1:
        mid = (lower + upper) >> 1
        running_p = mid * cst
        if running_p < p : 
            lower = mid
        else:
            upper = mid
    if upper != 10 ** c:
        p_thre = '0.' + str(upper)[:20]
    else:
        p_thre = '1.0'
    return float(p_thre)

def compute_rho(r : int, d : int, k : int, alpha : float, beta : float, delta_times_100 : int, search_exponent : int) -> int:
    """
    Wrapper to call all functions to compute rho
    """
    normalized = compute_rho_normalized(r, d, k, alpha, beta, delta_times_100)
    return return_to_base(normalized, search_exponent, k, d)

if __name__ == "__main__":
    # parse args : r, d, k 
    parser = create_arg_parse()
    args = parser.parse_args()
    logger = logging.getLogger()

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    compute_rho(args.radius, args.dimension, args.k, args.alpha, (1-args.alpha)/args.k, args.delta_times_100, args.search_exponent)