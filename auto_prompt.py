#!/usr/bin/env python3
import logging
import argparse
from prompt import sample
from common import dataset
from question import Question
from compute_params import auto_params

def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    
    logger = logging.getLogger("__prompt__")

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)

    ### Prompt
    logger.debug("Reached generation loop")
    radius = 2

    for i, row in enumerate(dataset):
        skip = False
        if (i >= 10):
            break
        
        logger.debug(f"Current question : {row['question']}")
        frag_filename = "question_prompt"+str(i)
        params = auto_params(row['question'], radius)

        current_question = Question(row['question'], row['answer']['normalized_aliases'], row['question_id'])
        current_question.generate_synonyms_albert(params["k"])

        # first N sample for smooth algorithm
        first_sample_name = frag_filename + "_1"
        sample(current_question, params["N"], params["k"], params["alpha"], first_sample_name)  

        # second N sample for smooth algorithm
        second_sample_name = frag_filename + "_2"
        sample(current_question, params["N"], params["k"], params["alpha"], second_sample_name)  

        # first m sample for certify algorithm
        third_sample_name = frag_filename + "_3"
        sample(current_question, params["m"], params["k"], params["alpha"], third_sample_name)
