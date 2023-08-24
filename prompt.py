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
    parser.add_argument("-m", "--num_lines", help="number of original questions to be taken from dataset, indexed from 0",type=int, default = 10)
    parser.add_argument("-r", "--max_radius", help="maximum radius to be certified",type=int, default=5)
    parser.add_argument("-N", "--smoothing_number", help="number of perturbations to be certified per radius",type=int, default=100)
    parser.add_argument("-t", "--top", help="number of synonyms to be considered for smoothing",type=int, default=10)
    parser.add_argument("-o", "--output", help="output file name", default="output.csv")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser
    
def certify_radius(current_question : Question,  N : int, top : int, filename : str):
    """
    For each radius :  
    * Generates <nb_pert> perturbed questions
    * For each perturbed question, generates N smoothed questions
    * Prompts the model (T5) on each question
    * Dumps the output in a file
    Parameters : 
    - current_question : Original question in dataset
    - nb_pert          : Number of perturbed questions to be generated for each original question
    - N                : Number of smoothed questions to be generated for each perturbed question
    - top              : number of synonyms considered when smoothing (variable "K" in the paper)
    - alpha            : Probability of not changing the original word when smoothing
    - filename         : output file name
    """
    with open(filename, "a") as f:
        torch.cuda.empty_cache() 
        writer = csv.writer(f)
        current_question.generate_synonyms_albert(top)
        smooth_prompts = current_question.smoothN(N, top, alpha)

        for _, smooth_prompt in tqdm.tqdm(enumerate(smooth_prompts)):

            
            input_ids = config.t5_tok(smooth_prompt, return_tensors="pt").input_ids
            gen_output = config.t5_qa_model.generate(input_ids)[0]
            smooth_answer = config.t5_tok.decode(gen_output, skip_special_tokens=True)
            
            del input_ids
            torch.cuda.empty_cache() 

            writer.writerow([current_question.id_num, smooth_prompt, smooth_answer])
        f.close()

if __name__ == "__main__":
    parser = create_arg_parse()
    args = parser.parse_args()

    alpha = args.alpha
    max_radius = args.max_radius
    N = args.smoothing_number
    top = args.top
    filename = args.output
    logger = logging.getLogger()
    if not args.verbose:
        logger.setLevel(logging.ERROR)
    elif args.verbose == 1:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)
    logging.debug(f"Alpha : {alpha}, max_radius : {max_radius}, N : {N}, top : {top}, filename : {filename}, verbose : {args.verbose}") 

    ### Load questions and answers
    logging.debug("Loading dataset")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    logging.debug("Loaded dataset")
    
    ### Create csv file
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "question",  "answer"])
        f.close()
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
        #Should be deleted after
        current_question = Question(row['question'], row['answer']['normalized_aliases'], row['question_id'])
        certify_radius(current_question, N, top, filename)
    

