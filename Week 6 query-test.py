from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Full list of SmoL models at link below - you may be able to use 1.7B depending on your machine
# https://huggingface.co/collections/HuggingFaceTB/smollm2

# the 360M model below is very small and may give weird results.

#model = 'HuggingFaceTB/SmolLM2-360M'
#model = 'HuggingFaceTB/SmolLM2-360M-Instruct'
model = 'microsoft/Phi-4-mini-instruct'
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)
#system_message = 'You are a helpful assistant. Give short answers.'
#system_message = 'You are a helpful French assistant who only answers in French'
system_message = 'You are an unhelpful assistant. Give short, bu comically wrong answers. '
instruction = 'What is the capital of France?'
prompt = f'<SYS> {system_message} <INST> {instruction} <RESP> '

response = pipeline(
   prompt, 
   max_length=200,
   repetition_penalty=1.05
)

print(response[0]['generated_text'])