from perturbations import *

if __name__ == "__main__":
    sentence = "Who let the dogs out?"
    perturbSentence(sentence, ["woof", "woof woof", "woof woof woof woof"], ["AAAAA", "BBBBB", "CCCCC", "DDDDD", "EEEEE"], 5, 1, samples_per_radius=5)
