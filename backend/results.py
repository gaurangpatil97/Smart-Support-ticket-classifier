import pandas as pd
import os

os.makedirs('results', exist_ok=True)

results = {
    "model": [
        "Original Keras Model (dirty data)",
        "TF-IDF + Logistic Regression",
        "Sentence Embeddings + Logistic Regression",
        "GPT-4o mini Zero-shot",
        "GPT-4o mini Few-shot",
        "DistilBERT + LoRA Model A (EOL dropped)",
        "DistilBERT + LoRA Model B (EOL included)",
        "Sentence Embeddings + LR (Synthetic Augmented)",
    ],
    "notebook": [
        "code.py (provided with dataset)",
        "01_baselineModel.ipynb",
        "01_baselineModel.ipynb",
        "03_prompt_classifier.ipynb",
        "03b_gpt_fewshot.ipynb",
        "LoraFine_tuning.ipynb (Colab)",
        "LoraFine_tuning.ipynb (Colab)",
        "01_baselineModel.ipynb (augmented)",
    ],
    "train_samples": [
        1572,
        962,
        962,
        "N/A",
        "N/A",
        928,
        900,
        2016,
    ],
    "test_macro_f1": [
        0.72,
        0.69,
        0.74,
        0.64,
        0.71,
        0.70,
        0.11,
        0.72,
    ],
    "test_accuracy": [
        0.79,
        0.75,
        0.78,
        0.67,
        0.72,
        0.78,
        0.30,
        0.78,
    ],
    "real_ticket_inference_48": [
        "N/A",
        "N/A",
        0.83,
        0.71,
        0.68,
        0.69,
        "N/A",
        0.73,
    ],
    "custom_50_sentences": [
        "N/A",
        "N/A",
        0.59,
        0.80,
        0.86,
        0.48,
        "N/A",
        0.84,
    ],
    "notes": [
        "Trained on dirty data with 610 duplicates. EOL F1 1.00 is fake — 44 identical templates. Severe overfitting after epoch 15.",
        "Honest baseline on deduplicated data. EOL dropped (2 samples unusable).",
        "Best overall on real tickets. Multilingual model handles short and non-English tickets better than TF-IDF.",
        "No training needed. Wins on clean sentences but struggles with noisy real tickets and vague Support general label.",
        "Few-shot examples improved Support general F1 from 0.55 to 0.66 but hurt Active Directory. Worse on real tickets than zero-shot.",
        "LoRA fine-tuned DistilBERT, EOL dropped, 6 classes. Only 0.54% of parameters trained. Matches accuracy of embeddings baseline.",
        "LoRA fine-tuned DistilBERT, EOL included, 7 classes. Complete failure — model predicted only 2 classes. EOL with 24 samples destabilized training.",
        "Balanced dataset with 336 samples per class using synthetic GPT tickets. Better on clean sentences but worse on real noisy tickets — distribution mismatch confirmed.",
    ]
}

df = pd.DataFrame(results)
df.to_csv('results/model_comparison.csv', index=False)
print(df.to_string(index=False))
print("\nSaved to results/model_comparison.csv")