import math
from compute_rho import *
def auto_params(question : str, radius : int, N : int = 1100, k : int = 3, alpha : float = 0.64, alpha_1 : float = 0.05, alpha_2 : float = 0.1, delta : int = 5):
    """
    Returns n >= 1000 (take 1100 for good measure)
    Returns param for prompt, smooth and certify
    """
    dimension = len(question.split())
    normalized = compute_rho_normalized(radius, dimension, k, alpha,  delta)
    rho = return_to_base(normalized, 50, k, dimension)
    if (rho > 0.98):
        quantile = -1
    quantile = int(1250*math.log(1/0.1))+200
    return {"radius" : radius, "N" : N, "k" : k, "alpha" : alpha, "alpha_1" : alpha_1, "alpha_2" : alpha_2, "delta" : delta, "m" : quantile}
