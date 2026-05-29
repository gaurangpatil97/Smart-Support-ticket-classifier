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

data = [
    # Fileservice (9)
    {"text": "I need access to the shared folder on the network drive", "true_label": "Fileservice"},
    {"text": "Please grant me read permission to the project files", "true_label": "Fileservice"},
    {"text": "I cannot access the file share from my workstation", "true_label": "Fileservice"},
    {"text": "Need to create a new shared folder for the finance team", "true_label": "Fileservice"},
    {"text": "My mapped network drive is showing disconnected", "true_label": "Fileservice"},
    {"text": "Please remove access to the HR shared drive for former employee", "true_label": "Fileservice"},
    {"text": "I need write access to the marketing documents folder", "true_label": "Fileservice"},
    {"text": "The shared drive is not visible after the network migration", "true_label": "Fileservice"},
    {"text": "Can you move my files from the old server to the new file share", "true_label": "Fileservice"},
    # Support general (9)
    {"text": "My computer is running very slow and I need help", "true_label": "Support general"},
    {"text": "I am unable to connect to the internet from my desk", "true_label": "Support general"},
    {"text": "The screen on my laptop has stopped working suddenly", "true_label": "Support general"},
    {"text": "I need a new keyboard my current one has broken keys", "true_label": "Support general"},
    {"text": "Please help me set up my new workstation", "true_label": "Support general"},
    {"text": "My headset is not being detected by the computer", "true_label": "Support general"},
    {"text": "I spilled water on my laptop and now it wont turn on", "true_label": "Support general"},
    {"text": "The monitor keeps flickering and is hard to work with", "true_label": "Support general"},
    {"text": "I need IT support to help me configure my work laptop", "true_label": "Support general"},
    # O365 (8)
    {"text": "I cannot login to my Outlook account since this morning", "true_label": "O365"},
    {"text": "Teams is not loading and I have a meeting in 10 minutes", "true_label": "O365"},
    {"text": "My OneDrive is not syncing files to the cloud", "true_label": "O365"},
    {"text": "I am not receiving any emails in my Outlook inbox", "true_label": "O365"},
    {"text": "SharePoint site is throwing an access denied error", "true_label": "O365"},
    {"text": "I need a license assigned to my account for Microsoft Teams", "true_label": "O365"},
    {"text": "Excel online is crashing when I open large spreadsheets", "true_label": "O365"},
    {"text": "My Office 365 subscription has expired and I cannot use Word", "true_label": "O365"},
    # Software (8)
    {"text": "Adobe Acrobat crashes every time I try to open a PDF", "true_label": "Software"},
    {"text": "I need to install the latest version of Chrome on my machine", "true_label": "Software"},
    {"text": "The accounting software is throwing a license error on startup", "true_label": "Software"},
    {"text": "AutoCAD is not responding after the recent Windows update", "true_label": "Software"},
    {"text": "Please install VPN client software on my new laptop", "true_label": "Software"},
    {"text": "I need Zoom installed and configured on my workstation", "true_label": "Software"},
    {"text": "The ERP system keeps logging me out every few minutes", "true_label": "Software"},
    {"text": "Can you update the antivirus software on my computer", "true_label": "Software"},
    # Active Directory (8)
    {"text": "My account has been locked out and I cannot log in", "true_label": "Active Directory"},
    {"text": "Please reset my domain password I have been locked out", "true_label": "Active Directory"},
    {"text": "I need a new user account created for a new team member", "true_label": "Active Directory"},
    {"text": "The new employee needs to be added to the sales security group", "true_label": "Active Directory"},
    {"text": "Please disable the account of the employee who left last week", "true_label": "Active Directory"},
    {"text": "I am getting invalid credentials error when logging into the domain", "true_label": "Active Directory"},
    {"text": "Need to update the display name and email for a user in Active Directory", "true_label": "Active Directory"},
    {"text": "My account does not have permission to access the admin tools", "true_label": "Active Directory"},
    # Computer-Services (8)
    {"text": "The office printer is not connecting to any computers on the floor", "true_label": "Computer-Services"},
    {"text": "I need a new laptop ordered for an incoming team member", "true_label": "Computer-Services"},
    {"text": "My desktop computer will not power on this morning", "true_label": "Computer-Services"},
    {"text": "The printer on the second floor is showing offline status", "true_label": "Computer-Services"},
    {"text": "I need more RAM installed in my workstation for video editing", "true_label": "Computer-Services"},
    {"text": "My docking station is not charging my laptop or connecting displays", "true_label": "Computer-Services"},
    {"text": "The conference room PC needs to be reimaged before the client visit", "true_label": "Computer-Services"},
    {"text": "Please schedule a hardware inspection for my machine it is overheating", "true_label": "Computer-Services"},
]

df = pd.DataFrame(data)
results = []

print("Running LoRA inference on 50 custom sentences...\n")
for _, row in df.iterrows():
    results.append(predict(row['text']))

df['predicted'] = results

print("===== INFERENCE RESULTS =====\n")
for _, row in df.iterrows():
    status = "✅" if row['true_label'] == row['predicted'] else "❌"
    print(f"{status}  True: {row['true_label']:<20} Predicted: {row['predicted']}")

correct = (df['true_label'] == df['predicted']).sum()
total = len(df)
print(f"\n===== SUMMARY =====")
print(f"Correct : {correct}/{total}")
print(f"Accuracy: {correct/total*100:.1f}%")
print("\n===== CLASS REPORT =====")
print(classification_report(df['true_label'], df['predicted']))