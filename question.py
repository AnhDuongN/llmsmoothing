#!/usr/bin/env python3
from random import randint, sample, random
from collections import defaultdict
import logging
import config
class Question:
    def __init__(self, question : str, answer : list, id_num : int):
        self.question = question
        self.questionWords = question.split()
        self.answer = answer
        self.id_num = id_num
        self.perturbations = defaultdict(list)
        self.synonyms = defaultdict(list)

    def __str__(self) -> str:
        return f"{self.id_num} \t {self.question} \t {self.answer}"
    
    def generate_synonyms_albert(self, top) -> dict:
        """
        Generates the synonyms of each word in the question, using ALBERT for Masked Language Modeling
        Parameters : 
        - top : number of synonyms to be generated for each word
        Returns : Dictionary indexed as <word, list of synonyms>
        """
        smoothing_dict = defaultdict(list)
        for i, word in enumerate(self.questionWords):
            if i == len(self.questionWords) -1:
                masked_question = ' '.join(self.questionWords[:-1])+"[MASK]?"
            else:
                masked_question = ' '.join(self.questionWords[:i]) + "[MASK]" + ' '.join(self.questionWords[i+1:])
            synonyms = config.smoothing_model(masked_question,top_k=top)
            for i, preds in enumerate(synonyms):
                smoothing_dict[word].append(preds["token_str"].replace('_',''))
        self.synonyms = smoothing_dict
    
    def generate_synonyms_glove(self, top) -> dict:
        """
        Generates the synonyms of each word in the question, using GloVe for Masked Language Modeling
        Parameters : 
        - top : number of synonyms to be generated for each word
        Returns : Dictionary indexed as <word, list of synonyms>
        """
        return

    def smoothN(self, N : int, top : int, alpha : float) -> list:
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
    
    def generatePerturbedQuestion(self, radius : int, max_turns : int = 10) -> str:
        """
        From a question, generates a perturbed questions with <radius> word substitutions
        Parameters : 
        - radius        : number of word perturbations
        - max_turns     : max number of re-draws to find a new perturbation distribution. 
                        If the number of re-draws exceeds max_turns, stop re-drawing and 
                        return no perturbation.
        Returns : str : A perturbed question
        """
        words_list = self.question.split()
        indices, replacements = self.generateSample(radius, max_turns)

        for i, (index, replacement) in enumerate(zip(indices, replacements)):
            words_list[index] = list(config.vocab.keys())[replacement]

        return PerturbedQuestion(' '.join(words_list).replace("?", "").replace("_","")+'?', self.answer, self.id_num, radius)
    
    def generateSample(self, radius : int, max_turns : int):
        """
        Generates a tuple containing the list of the indices of words to be perturbed, 
        and the index of the word in the tokenizer vocab to be substituted in.
        Parameters : 
        - radius        : number of word perturbations
        - max_turns     : max number of re-draws to find a new perturbation distribution. 
                        If the number of re-draws exceeds max_turns, stop re-drawing and 
                        return no perturbation.
        Returns : list[int], list[int] : list of word index to be perturbed, and for each word,
                                        the index of the word to be substituted in the tokenizer vocab.
        """
        perturbedIndex, replaceIndex = [], [None]*radius
        turns = 0
        while True: 
            if turns == max_turns:
                logging.info(f"Generation of perturbations exceded {max_turns} rolls")
                return [], []
            perturbedIndex = sample(range(0, len(self.questionWords)), radius)
            replaceIndex = [None]*radius
            for i, _ in enumerate(replaceIndex):
                replaceIndex[i] = randint(0, config.vocab_size-1)
            if (perturbedIndex, replaceIndex) not in self.perturbations[radius]:
                break
            turns +=1
        self.perturbations[radius].append((perturbedIndex, replaceIndex))
        return perturbedIndex, replaceIndex
    

class PerturbedQuestion(Question):
    def __init__(self, question, answer, id_num, radius):
        super().__init__(question, answer, id_num)
        self.radius = radius
    
    def __str__(self):
        return f"Perturbed question : {self.question}, original question id : {self.id_num}, radius : {self.radius}"
