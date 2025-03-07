from transformers import AutoModelForCausalLM, AutoTokenizer
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# Generate text
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=50)
# Print the result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))