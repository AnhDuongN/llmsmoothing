import argparse
from smooth import smooth
import logging
import numpy as np
import csv


if __name__ == "__main__":
    """
    Smoothing step (Algorithm 2)
    Outputs a file, such that for each question, the "center" answer is given, and "ABSTAINED FROM ANSWERING" is given
    if smoothing is not possible.
    """
    logger = logging.getLogger("__smooth__")
    logger.setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()

    N = 1100
    alpha_1 = 0.05
    delta = 0.05
    delta_1 = 2*np.exp(-2*N*(alpha_1**2)) #see theorem 3

    if True:
        from common import dataset

    logger.debug("Reached generation loop")
    
    with open("smooth.csv", "w") as f:
        writer = csv.writer(f)
        for i, row in enumerate(dataset):

            frag_filename = "question_prompt"+str(i)
            first_sample_name = frag_filename + "_1"
            second_sample_name = frag_filename + "_2"
                
            writer.writerow([row['question'], smooth(delta, delta_1, first_sample_name, second_sample_name)])
        f.close()
    

