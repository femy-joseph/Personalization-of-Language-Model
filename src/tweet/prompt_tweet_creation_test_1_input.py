import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

prompt = "Paraphrase the following tweet without any explanation before or after it: Although I have a lot to do today, I am currently feeling unoccupied."

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Generate prompt using the model
input_ids = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids,max_length=100,max_new_tokens=20)

    # Decode and print the generated prompt
generated_prompt = tokenizer.decode(outputs.reshape(-1))

print(generated_prompt)
