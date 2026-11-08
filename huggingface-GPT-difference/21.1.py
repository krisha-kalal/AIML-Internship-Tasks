import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# load pre-trained GPT-2 tokenizer and causal language model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# set the padding token
tokenizer.pad_token = tokenizer.eos_token

# prepare text input prompt
prompt = "Deep learning and transformer architectures have completely revolutionized"
inputs = tokenizer(prompt, return_tensors="pt")

# generate text (forward pass using top-p / nucleus sampling for creative, natural output)
with torch.no_grad():
    output_tokens = model.generate(
        **inputs,
        max_length=60,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.7,
        no_repeat_ngram_size=2
    )

# decode token sequences back into readable string format
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
