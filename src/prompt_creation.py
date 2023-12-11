from transformers import T5Tokenizer, T5ForConditionalGeneration
#document_content = """Since the Penn State scandal, questions about protecting children from sexual abuse seem to be on parents' minds all the time."""

#question = "Generate a headline for the following article:"
article = "article - Here are a few tips to keep your teen safe when using the Internet and other web-based technologies. If you think it's an awkward conversation; you can hand them this blog to read."
#input = question + article
profile_article = "example article - Although the age of social media has dramatically lowered the threshold on privacy standards, many adults are still reticent about reporting their suspicions about child abuse and neglect."
profile_title = "example title - If You See Something, Please Do Something to Prevent Child Abuse"
question2 = "Consider the example article and example title inorder to generate a headline for the following article:"
profile = question2 + article + profile_article + profile_title


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Tokenize the document content
tokenized_text = tokenizer.encode(input, return_tensors="pt")

# Example: Generating a question prompt from the document content
prompt =   question2 + profile

# Generate prompt using the model
input_ids = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(input_ids)

# Decode and print the generated prompt
generated_prompt = tokenizer.decode(outputs.reshape(-1))
print(generated_prompt)