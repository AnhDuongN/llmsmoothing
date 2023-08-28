import common
import pandas as pd
import compute_rho
import numpy as np
import csv

def certify(filename : str, center : str) -> list[float]:
    data = pd.read_csv(filename)
    answers = data['answer'].tolist()
    radii = [None]*len(answers)
    for i in range(len(answers)):
        radii[i] = common.compute_wmd(center, answers[i])
    return radii

def fin_certify(filename : str, center : str,  r : int, d : int, k : int, alpha : float, 
                delta_times_100 : int, search_exponent : int, alpha_2 : float, m : int) -> float:
    """
    """
    beta = (1-alpha)/k
    p = compute_rho.compute_rho(r, d, k, alpha, beta, delta_times_100, search_exponent)
    radii = certify(filename, center)
    q = p + np.sqrt(np.log(1/alpha_2)/(2*m))
    return (1+beta)*np.quantile(radii, q)

if __name__ == "__main__":
    with open("output.csv", "w") as f:
        writer = csv.writer(f)
        with open("smooth.csv", "r") as g:
            next(g)
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                question, center = row[0], row[1]
                radius = fin_certify("question"+str(i)+"_3", center, r, len(question.split()), k, 
                            alpha, delta_times_100, search_exponent, alpha_2, m)
                f.writerow([question, center, radius])
            g.close()
        f.close()