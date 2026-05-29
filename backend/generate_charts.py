import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sentence_transformers import SentenceTransformer
import os

os.makedirs('static/images', exist_ok=True)

test = pd.read_csv('data/test_final.csv')
X_te = test['text_clean']
y_te = test['label']

st_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X_te_emb = st_model.encode(X_te.tolist(), show_progress_bar=True)

def plot_cm(y_true, y_pred, title, filename):
    classes = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'static/images/{filename}', dpi=150)
    plt.close()
    print(f"Saved {filename}")

# TF-IDF
with open('models/tfidf_pipeline.pkl', 'rb') as f:
    tfidf = pickle.load(f)
y_pred_tfidf = tfidf.predict(X_te)
plot_cm(y_te, y_pred_tfidf, 'TF-IDF + LR Confusion Matrix', 'tfidf_confusion_matrix.png')

# Embeddings baseline
with open('models/emb_classifier.pkl', 'rb') as f:
    emb = pickle.load(f)
y_pred_emb = emb.predict(X_te_emb)
plot_cm(y_te, y_pred_emb, 'Sentence Embeddings + LR Confusion Matrix', 'emb_confusion_matrix.png')

# Augmented embeddings
with open('models/emb_augmented_classifier.pkl', 'rb') as f:
    emb_aug = pickle.load(f)
y_pred_aug = emb_aug.predict(X_te_emb)
plot_cm(y_te, y_pred_aug, 'Augmented Embeddings + LR Confusion Matrix', 'emb_augmented_confusion_matrix.png')

# GPT zero-shot
gpt_zero = pd.read_csv('results/gpt_test_predictions.csv')
plot_cm(gpt_zero['label'], gpt_zero['predicted'], 'GPT-4o mini Zero-shot Confusion Matrix', 'gpt_zeroshot_confusion_matrix.png')

# GPT few-shot
gpt_few = pd.read_csv('results/gpt_fewshot_predictions.csv')
plot_cm(gpt_few['label'], gpt_few['pred_few_shot'], 'GPT-4o mini Few-shot Confusion Matrix', 'gpt_fewshot_confusion_matrix.png')

print("\nAll confusion matrices saved to static/images/")