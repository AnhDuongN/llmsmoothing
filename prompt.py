#!/usr/bin/env python3
import random
import logging
import timeit
import csv
from question import Question
from perturbed_question import PerturbedQuestion
from collections import defaultdict
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from datasets import load_dataset, get_dataset_split_names

logging.basicConfig(level=logging.DEBUG)

def certify_radius(current_question : Question, nb_pert : int,  N : int, top : int, radius : int):
    #TODO : description
    global t5_tok, t5_qa_model
    with open("output.csv", "a") as f:
        writer = csv.writer(f)
        for j in range(nb_pert):
            prompt = current_question.generatePerturbedQuestion()
            prompt.generate_synonyms(top)
            smooth_prompts = prompt.smoothN(N,top)
            for _, smooth_prompt in enumerate(smooth_prompts):
                start = timeit.timeit()
                input_ids = t5_tok(smooth_prompt, return_tensors="pt").input_ids
                gen_output = t5_qa_model.generate(input_ids)[0]
                smooth_answer = t5_tok.decode(gen_output, skip_special_tokens=True)
                end = timeit.timeit()
                print(f"Inference time : {end-start}")
                writer.writerow([current_question.id_num, smooth_prompt, radius, smooth_answer])
        f.close()

if __name__ == "__main__":
    ### Argparse these things ###
    perturbations_per_radius = 10
    alpha = 0.85
    num_lines = 1
    max_radius = 5
    N = 10
    top = 5

    ### Load Q/A model and vocab ###

    t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-11b-ssm-tqa")
    t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")
    logging.debug("Loaded t5 model")

    vocab = t5_tok.get_vocab()
    vocab_size = t5_tok.vocab_size

    ### Load questions and answers
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    logging.debug("Loaded dataset")

    ### Load smoothing model
    smoothing_model = pipeline('fill-mask', model='albert-base-v2')

    ### Create csv file
    with open("output.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["question_id", "perturbed_question", "radius", "answer"])
        f.close()
    
    ### Prompt
    for i, row in enumerate(dataset):
        current_question = Question(row['question'], row['answer']['normalized_aliases'], row['question_id'], vocab_size, vocab)
        for j in range(1, max_radius):
            certify_radius(perturbations_per_radius, N, top, j)
    

