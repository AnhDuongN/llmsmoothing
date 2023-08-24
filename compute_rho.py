### outer for loop here
###
import math
def compute_cardinality(u : int, v : int, r : int, d : int, k : int) -> int:
    """
    Computes Lemma 11
    """
    cardinality = 0
    first_i = max(0, v-r)
    end_i = min(u, d-r, (u+v-r)//2)
    if (k < i):
        return -1
    for i in range(first_i, end_i+1):
        j = u + v - 2*i - r 
        cardinality += ((k-1)**j * math.factorial(r) * k**i * math.factorial(d-r)) \
        / (math.factorial(u-i-j) * math.factorial(v-i-j) * math.factorial(j) * math.factorial(d-r-i) * math.factorial(i))
    return cardinality

def compute_all_cardinalities(r : int , d : int, k : int) -> int:
    cardinalities_list = []
    for u in range(1, d+1):
        for v in range(u+1, d+1):
            cardinalities_list.append(((u, v), compute_cardinality(u,v,r,d,k)))
    cardinalities_list.sort(key = lambda x : x[1])
    return cardinalities_list
