from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")
t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-11b-ssm-tqa", device_map="auto")

enc = t5_tok(
    "This is a meaningless test sentence to show how you can get word embeddings", return_tensors="pt", return_attention_mask=False, add_special_tokens=False
)

output = t5_qa_model.encoder(
    input_ids=enc["input_ids"], 
    attention_mask=enc["attention_mask"], 
    return_dict=True
)
print(output.last_hidden_state)