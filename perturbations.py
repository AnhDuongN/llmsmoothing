#!/usr/bin/env python3
import random
import logging
from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_split_names

logging.basicConfig(level=logging.DEBUG)

def perturbSentence(sentence : str, answers : list,  id : str,  samples_per_radius : int = 10):
    """
    From the original sentence, generates samples_per_radius perturbed sentences for radii from 1 to the length of the sentence
    (i.e. all words are changed).
    Arguments : 
    - sentence
    - vocab_size         : Size of the vocabulary of the Language Model that is going to be prompted
    - samples_per_radius : Number of perturbed sentences to be generated for each perturbation size
    - id                 : order of the initial sentence in input file
    Output : 
    """
    global vocab, vocab_size
    logging.debug(f"Generating perturbations for {sentence}")
    words_list = sentence.split()
    for radius in range(1, len(words_list)):
        logging.debug(f"Radius = {radius}")
        for i in range(samples_per_radius):
            indices, replacements = perturbationSample(len(words_list), radius, vocab_size)
            logging.debug(f"Indices : {indices}, Replacements : {replacements}")
            for i, (index, replacement) in enumerate(zip(indices, replacements)):
                words_list[index] = vocab[replacement]
            perturbed_sentence = ' '.join(words_list).replace("?", "")
            perturbed_sentence = perturbed_sentence+"?"
            #compute rho
            #call smooth
            #call certify

    logging.debug("Finished generating perturbations for {sentence}")


def perturbationSample(length : int, radius : int, vocab_size : int): 

    logging.debug(f"Generating perturbations : length = {length}, radius = {radius}")
    perturbedIndex = random.sample(range(0,length), radius)
    replaceIndex = [None]*radius
    for i, _ in enumerate(replaceIndex):
        replaceIndex[i] = random.randint(0, vocab_size-1)
    return perturbedIndex, replaceIndex


    
if __name__ == "__main__()":
    t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")
    print("Loaded tokenizer!")
    vocab = t5_tok.get_vocab()
    vocab_size = t5_tok.vocab_size

    # for sentence in input file
    # call perturb sentence
    
    # logging.debug("Loading model : RoBERTa")
    # roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    # logging.debug("Roberta loaded")
    # print(roberta.fill_mask('The first Star wars movie came out in <mask>', topk=10))
    # dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")

    

