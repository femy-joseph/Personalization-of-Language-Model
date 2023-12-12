import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

with open("/Users/priyankavirupakshappadevoor/CS_646_project/src/eval/output.txt","r") as f:
    prompts = f.readlines()

#prompt = "Generate a headline for the following article Here are a few tips to keep your teen safe when using the Internet and other web based technologies If you think it s an awkward conversation you can hand them this blog to read here is the example article query pairThe New York Society for the Prevention of Cruelty to Children (NYSPCC), the first child protection agency in the world, wants all children to have a safe and happy summer. Here are our tips for keeping children safe.Summer Safety Tips for Your Children"

data = {
  "task": "LaMP_4",
  "golds": [
  ]
}
itr = 0
for prompt in prompts:
    itr+=1
    print("iteration:",itr)
    output_dict = {}
    prompt = prompt.split("-")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Generate prompt using the model
    input_ids = tokenizer.encode(prompt[1], return_tensors="pt")
    outputs = model.generate(input_ids)

    # Decode and print the generated prompt
    generated_prompt = tokenizer.decode(outputs.reshape(-1))
    output_dict['id'] = prompt[0]
    output_dict['output'] = generated_prompt
    data['golds'].append(output_dict)
print(data)
with open("preds_json.json", "w") as f:
  json.dump(data, f)