import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

classes = ['Fileservice', 'Support general', 'O365', 'Software', 'Active Directory', 'Computer-Services']

def classify_zero_shot(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"""You are an IT helpdesk ticket classifier.
Classify the following support ticket into exactly one of these categories:
{', '.join(classes)}

Ticket: {text}

Reply with only the category name, nothing else."""}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

test = pd.read_csv('data/test_final.csv')
test = test.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 8), random_state=42)
).reset_index(drop=True)

print(f"Sampled {len(test)} tickets")
print(test['label'].value_counts())

results = []
print(f"\nRunning GPT inference on {len(test)} test samples...\n")
for i, row in test.iterrows():
    try:
        pred = classify_zero_shot(row['text_clean'])
        results.append(pred)
        if i % 10 == 0:
            print(f"{i}/{len(test)} done")
        time.sleep(0.1)
    except Exception as e:
        print(f"Error at {i}: {e}")
        results.append("Error")

test['predicted'] = results

print("\n===== INFERENCE RESULTS =====\n")
for _, row in test.iterrows():
    status = "✅" if row['label'] == row['predicted'] else "❌"
    print(f"{status}  True: {row['label']:<20} Predicted: {row['predicted']}")

correct = (test['label'] == test['predicted']).sum()
total = len(test)
print(f"\n===== SUMMARY =====")
print(f"Correct : {correct}/{total}")
print(f"Accuracy: {correct/total*100:.1f}%")

print("\n===== CLASS REPORT =====")
print(classification_report(test['label'], test['predicted']))

test.to_csv('results/gpt_test_predictions.csv', index=False)
print("\nPredictions saved to results/gpt_test_predictions.csv")