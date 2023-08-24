#!/usr/bin/env python3
import config
import logging
import timeit
import csv
import argparse
import tqdm
import torch
from question import Question
from datasets import load_dataset


def create_arg_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", help="alpha",type=float, default=0.75)
    parser.add_argument("-n", "--num_lines", help="number of original questions to be taken from dataset, indexed from 0",type=int, default = 10)
    parser.add_argument("-r", "--max_radius", help="maximum radius to be certified",type=int, default=5)
    parser.add_argument("-N", "--smoothing_number", help="number of perturbations to be certified per radius",type=int, default=100)
    parser.add_argument("-m", "--quartile", type=int, default=100)
    parser.add_argument("-t", "--top", help="number of synonyms to be considered for smoothing",type=int, default=10)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser

def sample(current_question : Question,  N : int, top : int, alpha : float, filename : str):
    """
    For each radius :  
    * Generates N smoothed questions
    * Prompts the model (T5) on each question
    * Dumps the output in a file
    Parameters : 
    - current_question : Original question in dataset
    - N                : Number of smoothed questions to be generated for each perturbed question
    - top              : number of synonyms considered when smoothing (variable "K" in the paper)
    - alpha            : Probability of not changing the original word when smoothing
    - filename         : output file name
    """
    with open(filename, "w") as f:
        torch.cuda.empty_cache() 
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        smooth_prompts = current_question.generate_smooth_N_questions(N, top, alpha)

        for _, smooth_prompt in tqdm.tqdm(enumerate(smooth_prompts)):
            
            input_ids = config.t5_tok(smooth_prompt, return_tensors="pt").input_ids
            gen_output = config.t5_qa_model.generate(input_ids)[0]
            smooth_answer = config.t5_tok.decode(gen_output, skip_special_tokens=True)
            
            del input_ids
            torch.cuda.empty_cache() 

            writer.writerow([smooth_prompt, smooth_answer])
        f.close()

if __name__ == "__main__":
    """
    Samples Z = {z_i} such that z_i follows a distribution f(\phi (x)) three times, twice for "smooth" step and once for "certify" step.
    For each question, outputs 3 files corresponding to the three Z samples.
    Computation of MEB and of the center of the ball is done in smooth.py.
    Computation of the max radius is done in certify.py
    """
    parser = create_arg_parse()
    args = parser.parse_args()

    alpha = args.alpha
    max_radius = args.max_radius
    N = args.smoothing_number
    top = args.top
    m = args.quartile

    logger = logging.getLogger()

    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    logging.debug(f"Alpha : {alpha}, max_radius : {max_radius}, N : {N}, top : {top}, verbose : {args.verbose}") 

    ### Load questions and answers
    logging.debug("Loading dataset")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    logging.debug("Loaded dataset")
    
    ### Prompt
    if args.num_lines : 
        num_lines = args.num_lines
    else:
        num_lines = len(dataset)

    logging.debug("Reached generation loop")
    
    for i, row in enumerate(dataset):
        if (i >num_lines):
            break
        logging.debug(f"Current question : {row['question']}")
        frag_filename = "question"+str(i)
        #Should be deleted after
        current_question = Question(row['question'], row['answer']['normalized_aliases'], row['question_id'])
        current_question.generate_synonyms_albert(top)

        first_sample_name = frag_filename + "_1"
        sample(current_question, N, top, alpha, first_sample_name)
        second_sample_name = frag_filename + "_2"
        sample(current_question, N, top, alpha, second_sample_name)
        third_sample_name = frag_filename + "_3"
        sample(current_question, m, top, alpha, third_sample_name)
    

