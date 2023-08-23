import logging
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch

### Load Q/A model and vocab ###
assert(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logging.debug("Loading model and tokenizer")
t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-11b-ssm-tqa")
t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")
logging.debug("Loaded t5 model and tokenizer")

vocab = t5_tok.get_vocab()
vocab_size = t5_tok.vocab_size

### Load smoothing model
smoothing_model = pipeline('fill-mask', model='albert-base-v2', device=0)
