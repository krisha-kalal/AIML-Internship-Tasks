import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load a RoBERTa model fine-tuned for sentiment analysis
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# prepare text input
text = "This movie was absolutely fantastic and brilliant!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# run inference (forward pass)
with torch.no_grad():
    outputs = model(**inputs)
    
# convert raw scores (logits) into probabilities and get the highest label
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
predicted_class_id = torch.argmax(probabilities, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_id]
confidence_score = probabilities[0][predicted_class_id].item()

print(f"Text: '{text}'")
print(f"Sentiment: {predicted_label} ({confidence_score * 100:.2f}% confidence)")
