import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

# load model
with open('models/emb_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# load clean test data and sample stratified 48
test = pd.read_csv('data/test_final.csv')
sample = test.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 8), random_state=42)
).reset_index(drop=True)

print(f"Sampled {len(sample)} tickets")
print(sample['label'].value_counts())

# predict
X = st_model.encode(sample['text_clean'].tolist(), show_progress_bar=False)
preds = clf.predict(X)

# results
print("\n===== INFERENCE RESULTS =====\n")
for i, row in sample.iterrows():
    true = row['label']
    pred = preds[i]
    status = "✅" if true == pred else "❌"
    print(f"{status} True: {true:<20} Predicted: {pred}")

# summary
correct = sum(1 for t, p in zip(sample['label'], preds) if t == p)
total = len(sample)
print(f"\n===== SUMMARY =====")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct/total*100:.1f}%")

print("\n===== CLASS REPORT =====")
print(classification_report(sample['label'], preds))