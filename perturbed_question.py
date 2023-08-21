#!/usr/bin/env python3

from random import random, randint
from collections import defaultdict

class PerturbedQuestion:
    def __init__(self, question, id_num):
        self.question = question
        self.questionWords = question.split()
        self.id_num = id_num
        self.synonyms = defaultdict(list)
    
    def __str__(self):
        return f"Perturbed question : {self.question}, original question id : {self.id_num}"
    
    def smoothN(self, N : int = 25, top : int = 10, alpha = 0.85) -> list:
        """
        Generates N questions following a smoothing distribution around the perturbed question
        Parameters :
        - N          : number of questions to be generated
        - top        : number of synonyms considered when smoothing (variable "K" in the paper)
        - alpha      : probability of not smoothing a word by its synonym (variable "alpha" in the paper)
        Returns : List containing the N questions.
        """
        #Generate the list of synonyms of each word in the question once to compute smoothed-out questions more easily
        ret = []
        for i in range(N):
            ret.append(self.smooth(top, alpha))
        return ret

    def smooth(self, top : int, alpha : float) -> str:
        """
        Generates a question following a smoothing distribution around the perturbed question
        Parameters : 
        - top           : number of synonyms considered when smoothing (variable "K" in the paper)
        - alpha         : probability of not smoothing a word by its synonym (variable "alpha" in the paper)
        """

        new_question = self.question.split()
        for i, word in enumerate(new_question) :
            if random() > alpha:
                new_question[i] = self.synonyms[word][randint(0, top-1)]
        return ' '.join(new_question).replace('?','') + '?'

    def generate_synonyms(self, top : int = 10) -> dict:
        """
        Generates the synonyms of each word in the question, using ALBERT for Masked Language Modeling
        Parameters : 
        - top : number of synonyms to be generated for each word
        Returns : Dictionary indexed as <word, list of synonyms>
        """
        
        global smoothing_model
        smoothing_dict = defaultdict(list)
        for i, word in enumerate(self.questionWords):
            if i == len(self.questionWords -1):
                masked_question = ' '.join(self.questionWords[:-1])+"[MASK]?"
            else:
                masked_question = ' '.join(self.questionWords[:i]) + "[MASK]" + ' '.join(self.questionWords[i+1:])
            synonyms = smoothing_model(masked_question, top_k = top)
            for i, preds in enumerate(synonyms):
                smoothing_dict[word].append(preds["token_str"].replace('_',''))
        self.synonyms = smoothing_dict