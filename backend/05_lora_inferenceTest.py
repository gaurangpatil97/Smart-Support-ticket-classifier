import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report
import torch

adapter_path = "models/lora model a"
base_model_name = "distilbert-base-multilingual-cased"

label2id = {"Fileservice": 0, "Support general": 1, "Software": 2, "O365": 3, "Active Directory": 4, "Computer-Services": 5}
id2label = {0: "Fileservice", 1: "Support general", 2: "Software", 3: "O365", 4: "Active Directory", 5: "Computer-Services"}

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
print("Model loaded.")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2label[pred_id]

test = pd.read_csv('data/test_final.csv')
sample = test.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 8), random_state=42)
).reset_index(drop=True)

print(f"Sampled {len(sample)} tickets")
print(sample['label'].value_counts())

results = []
for _, row in sample.iterrows():
    results.append(predict(row['text_clean']))

sample['predicted'] = results

print("\n===== INFERENCE RESULTS =====\n")
for _, row in sample.iterrows():
    status = "✅" if row['label'] == row['predicted'] else "❌"
    print(f"{status}  True: {row['label']:<20} Predicted: {row['predicted']}")

correct = (sample['label'] == sample['predicted']).sum()
total = len(sample)
print(f"\n===== SUMMARY =====")
print(f"Correct : {correct}/{total}")
print(f"Accuracy: {correct/total*100:.1f}%")
print("\n===== CLASS REPORT =====")
print(classification_report(sample['label'], sample['predicted']))