from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "marianbasti/Llama-2-13b-fp16-alpaca-spanish"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text =""

prompt = "Eres un experto en lenguaje claro. Necesito que evalúes el siguiente texto en cuanto a claridad del lenguaje: "
prompt = prompt + text
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
