from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import timeit
t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-11b-ssm-tqa")
print("Loaded model!")
t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")
print("Loaded tokenizer!")

input_ids = t5_tok("When was Franklin D. Roosevelt born?", return_tensors="pt").input_ids
start = timeit.timeit()
gen_output = t5_qa_model.generate(input_ids)[0]
end = timeit.timeit()
print("Generation time = ", end-start, "Output : ")
print(t5_tok.decode(gen_output, skip_special_tokens=True))
