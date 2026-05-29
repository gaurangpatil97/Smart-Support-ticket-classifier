import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

# load augmented model
with open('models/emb_augmented_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)

st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# load stratified 48 real tickets
test = pd.read_csv('data/test_final.csv')
sample = test.groupby('label', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 8), random_state=42)
).reset_index(drop=True)

print(f"Sampled {len(sample)} tickets")
print(sample['label'].value_counts())

X = st_model.encode(sample['text_clean'].tolist(), show_progress_bar=False)
preds = clf.predict(X)

print("\n===== INFERENCE RESULTS =====\n")
for i, row in sample.iterrows():
    status = "✅" if row['label'] == preds[list(sample.index).index(i)] else "❌"
    print(f"{status}  True: {row['label']:<20} Predicted: {preds[list(sample.index).index(i)]}")

correct = sum(1 for t, p in zip(sample['label'], preds) if t == p)
total = len(sample)
print(f"\n===== SUMMARY =====")
print(f"Correct : {correct}/{total}")
print(f"Accuracy: {correct/total*100:.1f}%")
print("\n===== CLASS REPORT =====")
print(classification_report(sample['label'], preds))