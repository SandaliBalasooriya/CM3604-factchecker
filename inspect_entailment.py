import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

print("Label mapping:", model.config.id2label)

premise = "Maya enjoys playing the piano."
hypothesis = "Maya was born in London."

inputs = tokenizer(
    premise,
    hypothesis,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

print("Input shape:", inputs["input_ids"].shape)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

print("Logits:", logits)
print("Probabilities:", probabilities)

for index, probability in enumerate(probabilities[0].tolist()):
    label = model.config.id2label[index]
    print(f"{label}: {probability:.4f}")