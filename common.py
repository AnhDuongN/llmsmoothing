from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
import logging
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset

from numpy import zeros, float32 as REAL,double, sqrt, sum as np_sum

from gensim.corpora.dictionary import Dictionary
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from pyemd import emd

import logging
logger = logging.getLogger(__name__)

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
########## Load dataset #############
### Load questions and answers
logging.debug("Loading dataset")
dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
logging.debug("Loaded dataset")

#####################################
##### Word mover's distance utils ###

logging.debug("Loading WMD utils")

download('stopwords') 
stop_words = stopwords.words('english')
model = api.load('word2vec-google-news-300')

logging.debug("Loaded WMD utils!")

### COMPUTE Word Movers Distance ###
def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

def compute_wmd(sentence1 : str, sentence2 : str) -> float:
    """
    Wrapper to call wmdistance from gensim module. Computes word mover's distance using
    word2vec embeddings between the 2 input sentences.
    Parameters : 
    - document1 : first sentence
    - document2 : second sentence
    Returns : Word mover's distance between the 1st and 2nd sentences.
    """
    sentence_1 = preprocess(sentence1)
    sentence_2 = preprocess(sentence2)
    distance = model.wmdistance(sentence_1, sentence_2)
 #   if distance == float('inf'):
 #       print(f"Sentence 1 : {sentence1} \n Sentence 2 : {sentence2}")
    return distance

def encode(se : str):
    """
    Returns word encodings of the sentence. Used by alternate compute wmd function.
    Parameters : 
    - se : the sentence for which we need the word by word embeddings
    Returns : tuple(list[str], list[tensor(shape 1024)]) : list of words and list of respective embeddings
    """
    embedding = TransformerWordEmbeddings("google/t5-11b-ssm-tqa", subtoken_pooling='mean') # can be moved out of this function 
                                                                                            # to be used as a global variable
    sentence = Sentence(se)
    embedding.embed(sentence)
    return [token for token in sentence if token not in stop_words], [token.embedding for token in sentence if token not in stop_words]
            
def _compute_wmd(document1 : str, document2 : str) -> float:
    """
    Planned adaptation of WMD. 
    The problem with our initial way to compute Word Mover's Distance is that the model used to vectorize our words 
    and get their embeddings (Word2Vec) has a different vocabulary from our question-answer model (T5 - which uses 
    SentencePiece). A way to solve this problem is to use the word embeddings generated by SentencePiece, using 
    python-flair library. However, this is computationally costly and should be done on GPU.
    Parameters : 
    - document1 : first sentence
    - document2 : second sentence
    Returns : Word mover's distance between the 1st and 2nd sentences.
    !!! Disclaimer !!!
    This code copied from Gensim.WMD (with slight adaptations to use SentencePiece)
    """
    # TODO : test this on GPU
    document1, encodings1 = encode(document1)
    document2, encodings2 = encode(document2)

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)
    
    # Compute distance matrix.
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in docset1 or not t2 in docset2:
                continue
            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = sqrt(np_sum((encodings1[i] - encodings2[j])**2))

    if np_sum(distance_matrix) == 0.0:
        # `emd`gets stuck if the distance matrix contains only zeros.
        logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)

#####################

def verify_vocab_in_w2v(answer : str) -> bool:
    """
    Verifies if at least one word in the answer is in the vocabulary of the Word2Vec model
    so that Word Mover's Distance can be computed.
    Parameters : answer : the answer to verify
    Returns    : True if at least one word is in the vocabulary, otherwise False
    """
    answer = answer.split()
    for word in answer:
        if word in model.key_to_index:
            return True
    return False
