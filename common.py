from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
import logging
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch

##### Load Q/A model and vocab #####

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device used : {device}")

logging.debug("Loading model and tokenizer")
t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-11b-ssm-tqa", device_map="auto")
t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")
logging.debug("Loaded t5 model and tokenizer")

vocab = t5_tok.get_vocab()
vocab_size = t5_tok.vocab_size

#####################################
##### Load smoothing model ##########
smoothing_model = pipeline('fill-mask', model='albert-base-v2')
print("Loaded all models!")

#####################################
##### Word mover's distance utils ###

download('stopwords') 
stop_words = stopwords.words('english')
model = api.load('word2vec-google-news-300')

### COMPUTE Word Movers Distance ###
def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def compute_wmd(sentence1 : str, sentence2 : str) -> float:
    sentence_1 = preprocess(sentence1)
    sentence_2 = preprocess(sentence2)
    distance = model.wmdistance(sentence_1, sentence_2)
    return distance