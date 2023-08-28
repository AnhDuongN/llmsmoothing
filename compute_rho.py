import math
import csv
import argparse
import logging

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

def compute_likelihood_ratio(u : int, v : int, k : int, alpha : float, beta : float) -> float:
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
    alpha_tilda = alpha*100*k
    beta_tilda = beta*100*k
    return (alpha_tilda ** (v-u)) * (beta_tilda **(u-v))

def compute_cardinalities_and_ratios(r : int , d : int, k : int, alpha : float, beta : float) -> int:
    """
    Computes cardinalities and ratios for each tuple (u,v) in (0, d)^2.
    Parameters : 
    - r : perturbation radius
    - d : dimension (length of the vector)
    - k     : top k synonyms chosen to smooth each word (see e.g. Question.smoothN or see args)
    - alpha : probability that a word from the canonical sentence isn't substituted (same alpha as in Question.smoothN())
    - beta  : probability of each substitute if the word in the canonical sentence is substituted (same beta as in Question.smoothN())
    Returns : 
    A file (temp_card.csv) such that for each line the values are : 
    [u, v, cardinality of L(u;v;r), likelihood ratio n(x, bar(x))]
    """
    with open("temp_card.csv", 'w') as f : #this file is only temporary
        writer = csv.writer(f)
        for u in range(0, d+1):
            for v in range(u, d+1):
                writer.writerow([u, v, compute_cardinality(u,v,r,d,k), compute_likelihood_ratio(u,v,k,alpha,beta)])
                if v!= u: #divides by 2 the number of operations needed
                    writer.writerow([v, u, compute_cardinality(u,v,r,d,k), compute_likelihood_ratio(v,u,k,alpha,beta)]) 
        f.close()

def compute_rho_unusual(r : int, d : int, k : int, alpha : float, beta : float, delta_times_100 : int) -> int:
    """
    Idk what this is (some kind of binary search) but this works better than following the original algorithm so whatever
    """
    # Normalization is done to avoid floating point arithmetics (if the probabilities are too small, floats aren't as accurate)
    alpha_tilda = alpha*100*k
    beta_tilda = beta*100*k
    half_Z = (50+delta_times_100)*k * (100*k)**(d-1)

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

def sc(n : int):
    return "{:.2e}".format(n)

def return_to_base(p : int, c : int, k : int, d : int):
    upper = 10 ** c
    lower = 0
    cst = ((10*k) ** c) * ((100*k)**(d-c))
    logging.debug(f"Arguments : p : {p}")
    while upper-lower != 1:
        mid = (lower + upper) >> 1
        running_p = mid * cst
        logging.debug(f"running : mid : {sc(mid)}, cst : {sc(cst)}, running : {sc(mid *cst)} \n")
        logging.debug(f"Upper : {sc(upper)}, lower : {sc(lower)}, mid : {sc(mid)}, running : {sc(running_p)}, p : {sc(p)}")
        logging.debug(f"Types : Upper : {type(upper)}, lower : {type(lower)}, mid : {type(mid)}, running : {type(running_p)} \n \n")
        if running_p < p : 
            lower = mid
        else:
            upper = mid
    if upper != 10 ** c:
        p_thre = '0.' + str(upper)[:20]
    else:
        p_thre = '1.0'
    return p_thre


if __name__ == "__main__":
    # parse args : r, d, k 
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--radius", help="number of original questions to be taken from dataset, indexed from 0",type=int, default = 10)
    parser.add_argument("-d", "--dimension", help="delta (in paper)",type=int, default=14) #TODO better explanation
    parser.add_argument("-k", "--k", help="top_k", type = int, default = 10) 
    args = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)


    compute_cardinalities_and_ratios(args.radius, args.dimension, args.k, 0.75, 0.25)
    temp = compute_rho_unusual(args.radius, args.dimension, args.k, 0.75, 0.25/args.k, 5)
    print(temp)
    for i in range(10, 100, 10):
        print(f"Exponent : {i}, thresh : {return_to_base(temp, i, args.k, args.dimension)}")


    # do the sorting stuff
    # compute rho in 2 ways : like in paper and not like in paper
