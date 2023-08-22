#!/usr/bin/env python3
from random import randint, sample
from collections import defaultdict
import logging
from perturbed_question import PerturbedQuestion

class Question:
    def __init__(self, question : str, answer : list, id_num : int, vocab_size : int, vocab : dict):
        self.question = question
        self.length = len(question.split())
        self.answer = answer
        self.id_num = id_num
        self.perturbations = defaultdict(list)
        self.vocab = list(vocab.keys())
        self.vocab_size = vocab_size

    def __str__(self) -> str:
        return f"{self.id_num} \t {self.question} \t {self.answer}"
    
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
            words_list[index] = self.vocab[replacement]

        return PerturbedQuestion(' '.join(words_list).replace("?", "").replace("_","")+'?', self.id_num)
    
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
            perturbedIndex = sample(range(0, self.length), radius)
            replaceIndex = [None]*radius
            for i, _ in enumerate(replaceIndex):
                replaceIndex[i] = randint(0, self.vocab_size-1)
            if (perturbedIndex, replaceIndex) not in self.perturbations[radius]:
                break
            turns +=1
        self.perturbations[radius].append((perturbedIndex, replaceIndex))
        return perturbedIndex, replaceIndex
