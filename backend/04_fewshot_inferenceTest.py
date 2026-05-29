import pandas as pd
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

classes = ['Fileservice', 'Support general', 'O365', 'Software', 'Active Directory', 'Computer-Services']

few_shot_examples = {
    "Fileservice": [
        "I need access to the shared folder on the finance network drive",
        "Please grant read permission to the project files for my team"
    ],
    "Support general": [
        "My computer is running very slow and I need someone to help",
        "The monitor keeps flickering and is hard to work with"
    ],
    "O365": [
        "I cannot login to my Outlook account since this morning",
        "Teams is not loading and I have a meeting in 10 minutes"
    ],
    "Software": [
        "Adobe Acrobat crashes every time I try to open a PDF",
        "The accounting software is throwing a license error on startup"
    ],
    "Active Directory": [
        "My account has been locked out and I cannot log in to the domain",
        "Please reset my domain password I have been locked out"
    ],
    "Computer-Services": [
        "The office printer is not connecting to any computers on the floor",
        "My desktop computer will not power on this morning"
    ]
}

def classify_few_shot(text):
    examples_text = ""
    for label, examples in few_shot_examples.items():
        for ex in examples:
            examples_text += f"Ticket: {ex}\nCategory: {label}\n\n"

    prompt = f"""You are an IT helpdesk ticket classifier.
Here are some example tickets and their correct categories:

{examples_text}
Now classify this ticket into exactly one of these categories:
{', '.join(classes)}

Ticket: {text}

Reply with only the category name, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

test = pd.read_csv('data/test_final.csv')
sample = test.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 8), random_state=42)
).reset_index(drop=True)

print(f"Sampled {len(sample)} tickets")
print(sample['label'].value_counts())

results = []
print(f"\nRunning few-shot GPT on {len(sample)} tickets...\n")
for i, row in sample.iterrows():
    try:
        pred = classify_few_shot(row['text_clean'])
        results.append(pred)
        if i % 10 == 0:
            print(f"{i}/{len(sample)} done")
        time.sleep(0.1)
    except Exception as e:
        print(f"Error at {i}: {e}")
        results.append("Error")

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