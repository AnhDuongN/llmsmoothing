import math
import csv
import argparse
import logging


def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--radius", help="number of perturbations",type=int, default = 3)
    parser.add_argument("-d", "--dimension", help="length of input sentence",type=int, default=14) #TODO better explanation
    parser.add_argument("-k", "--k", help="number of synonyms to be considered for smoothing", type = int, default = 10) 
    parser.add_argument("-a", "--alpha", help="probability for a word not to be substituted in smoothing distribution",\
                        type=float, default=0.75)
    parser.add_argument("-D", "--delta_times_100", help="delta such that the fall of center hat(f) encloses 1/2 + delta probability \
                        mass of the smoothed f(x) TIMES 100", type=int, default=5)
    parser.add_argument("-c", "--search_exponent", help="search exponent for binary search",type=int, default = 30)
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
    - u           : coordinates flipped from x_c (the canonical sentence)
    - v           : coordinates flipped from bar(x_c) (the perturbed sentence)
    - k           : top k synonyms chosen to smooth each word (see e.g. Question.smoothN or see args)
    - alpha_tilda : probability that a word from the canonical sentence isn't substituted (same alpha as in Question.smoothN()) 
                    TIMES 100 TIMES K 
    - beta_tilda  : probability of each substitute if the word in the canonical sentence is substituted (same beta as in Question.smoothN()) 
                    TIMES 100 TIMES K
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

def compute_rho_normalized(r : int, d : int, k : int, alpha : float, delta_times_100 : int) -> int:
    """
    Computes a normalized (i.e. big integer) value of rho_r(p). We use large number normalization to avoid computing
    with small probabilities, which entails floating point arithmetics errors. This code is based from the implementation
    by the authors of [Tight certificates of adversarial robustness for randomly smoothed classifiers - Lee et al., 2019]

    Parameters : 
    - r               : Perturbation radius to certify
    - d               : Dimension - length of the sentence
    - k               : Number of synonyms considered at substitution in smoothing step
    - alpha           : probability that a word from the canonical sentence isn't substituted (same alpha as in Question.smoothN()) 
    - delta_times_100 : delta such that the fall of center hat(f) encloses 1/2 + delta probability mass of the smoothed f(x) TIMES 100
    Returns : Normalized value of rho
    """
    # Normalization is done to avoid floating point arithmetics (if the probabilities are too small, floats aren't as accurate)
    alpha_tilda = alpha*100*k
    beta_tilda = (1-alpha)*100 # [(1-alpha)/k] x 100 x k
    half_Z = (50+delta_times_100)*k * (100*k)**(d-1)

    compute_cardinalities_and_ratios(r, d, k, alpha_tilda, beta_tilda)

    with open("temp_card.csv", "r") as f:
        reader = csv.reader(f)
        cardinalities = list(reader)
    f.close()

    cardinalities = sorted(cardinalities, key=lambda x : x[-1], reverse=True)
    # At this state, each item in cardinalities is a list formatted as
    # [u, v, cardinality of L(u;v;r), likelihood ratio n(x, bar(x))]
    # sorted by decreasing likelihood ratios

    # The following is based on the code of the GitHub of the MIT guys from the l0 paper (see threshold_mnist.py)
    # This is an adaptation of algorithm 1 but with normalization to avoid floating point arithmetics errors
    p_given = 0
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

            # Binary search to estimate (0.5 - rho_r) / rho_prime_r.
            # We know it's upper bound is cardinality of L(u;v;r)

            while upper-lower != 1:
                mid = (lower+upper) >> 1
                running_Z = mid*q
                if running_Z < half_Z:
                    lower = mid
                else:
                    upper = mid
            p_given += upper*p
            return p_given
        
   # logger.critical("No p was computed!")
    return -1

def return_to_base(p : int, c : int, k : int, d : int) -> float:
    """
    Binary search to return the normalized rho value to the original interval [0,1)
    
    Parameters : 
    - p : computed rho value
    - c : search exponent for binary search
    - k : Number of synonyms considered at substitution in smoothing step
    - d : Dimension - length of the sentence
    Returns : Real rho value
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
        #logger.critical("Computed rho value is 1.")
    return float(p_thre)


if __name__ == "__main__":
    """
    Code for algorithm 1 (compute rho value).
    Standalone script to test the computation of rho. Normally, the functions in this module are 
    called by certify.py.
    """
    parser = create_arg_parse()
    args = parser.parse_args()
    logger = logging.getLogger("__compute_rho__")

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    normalized = compute_rho_normalized(args.radius, args.dimension, args.k, args.alpha,  args.delta_times_100)
    print(return_to_base(normalized, args.search_exponent, args.k, args.dimension))
