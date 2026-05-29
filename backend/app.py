import os
import pickle
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Setup Flask application and CORS
app = Flask(__name__)
CORS(app)

# Resolve directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ==========================================
# MODEL LOADERS
# ==========================================

# 1. Load TF-IDF Pipeline
tfidf_pipeline_path = os.path.join(MODELS_DIR, "tfidf_pipeline.pkl")
try:
    with open(tfidf_pipeline_path, "rb") as f:
        tfidf_pipeline = pickle.load(f)
    print("Successfully loaded TF-IDF pipeline.")
except Exception as e:
    print(f"Error loading TF-IDF pipeline: {e}")
    tfidf_pipeline = None

# 2. Load Sentence Embeddings Classifier
emb_classifier_path = os.path.join(MODELS_DIR, "emb_classifier.pkl")
try:
    with open(emb_classifier_path, "rb") as f:
        emb_classifier = pickle.load(f)
    print("Successfully loaded Sentence Embeddings classifier.")
except Exception as e:
    print(f"Error loading Sentence Embeddings classifier: {e}")
    emb_classifier = None

emb_augmented_classifier_path = os.path.join(MODELS_DIR, "emb_augmented_classifier.pkl")
try:
    with open(emb_augmented_classifier_path, "rb") as f:
        emb_augmented_classifier = pickle.load(f)
    print("Successfully loaded Augmented Sentence Embeddings classifier.")
except Exception as e:
    print(f"Error loading Augmented Sentence Embeddings classifier: {e}")
    emb_augmented_classifier = None

# Load SentenceTransformer model
try:
    st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("Successfully loaded SentenceTransformer model.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    st_model = None

# 3. Load DistilBERT + LoRA Model A
lora_path = os.path.join(MODELS_DIR, "lora model a")
base_model_name = "distilbert-base-multilingual-cased"
label2id = {
    "Fileservice": 0,
    "Support general": 1,
    "Software": 2,
    "O365": 3,
    "Active Directory": 4,
    "Computer-Services": 5
}
id2label = {v: k for k, v in label2id.items()}

try:
    lora_tokenizer = AutoTokenizer.from_pretrained(lora_path)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    lora_model.eval()
    print("Successfully loaded DistilBERT + LoRA model.")
except Exception as e:
    print(f"Error loading DistilBERT + LoRA model: {e}")
    lora_model = None
    lora_tokenizer = None

# 4. OpenAI Client
openai_key = os.getenv("OPENAI_API_KEY")
try:
    if openai_key:
        openai_client = OpenAI(api_key=openai_key)
        print("Successfully initialized OpenAI client.")
    else:
        print("WARNING: OPENAI_API_KEY not found in environment.")
        openai_client = None
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    openai_client = None


# ==========================================
# PREDICTION HELPER FUNCTIONS
# ==========================================

def predict_tfidf(text):
    if tfidf_pipeline is None:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        label = tfidf_pipeline.predict([text])[0]
        probs = tfidf_pipeline.predict_proba([text])[0]
        confidence = float(np.max(probs))
        return {"label": str(label), "confidence": confidence}
    except Exception as e:
        print(f"TF-IDF inference failed: {e}")
        return {"label": "unavailable", "confidence": 0.0}


def predict_embeddings(text):
    if emb_classifier is None or st_model is None:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        vector = st_model.encode([text], show_progress_bar=False)
        label = emb_classifier.predict(vector)[0]
        probs = emb_classifier.predict_proba(vector)[0]
        confidence = float(np.max(probs))
        return {"label": str(label), "confidence": confidence}
    except Exception as e:
        print(f"Sentence Embeddings inference failed: {e}")
        return {"label": "unavailable", "confidence": 0.0}


def predict_embeddings_augmented(text):
    if emb_augmented_classifier is None or st_model is None:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        vector = st_model.encode([text], show_progress_bar=False)
        label = emb_augmented_classifier.predict(vector)[0]
        probs = emb_augmented_classifier.predict_proba(vector)[0]
        confidence = float(np.max(probs))
        return {"label": str(label), "confidence": confidence}
    except Exception as e:
        print(f"Augmented Sentence Embeddings inference failed: {e}")
        return {"label": "unavailable", "confidence": 0.0}


def predict_gpt(text):
    if openai_client is None:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        messages = [
            {"role": "system", "content": "You are an IT support ticket classifier. Classify into exactly one of: Fileservice, Support general, Software, O365, Active Directory, Computer-Services. Reply with the class name only."},
            {"role": "user", "content": "I can't access the shared drive"},
            {"role": "assistant", "content": "Fileservice"},
            {"role": "user", "content": "My Outlook keeps crashing"},
            {"role": "assistant", "content": "O365"},
            {"role": "user", "content": "Need admin rights on my machine"},
            {"role": "assistant", "content": "Active Directory"},
            {"role": "user", "content": "Laptop won't boot after update"},
            {"role": "assistant", "content": "Computer-Services"},
            {"role": "user", "content": "How do I install Python?"},
            {"role": "assistant", "content": "Software"},
            {"role": "user", "content": "Not sure who to contact"},
            {"role": "assistant", "content": "Support general"},
            {"role": "user", "content": text}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        label = response.choices[0].message.content.strip()
        # Clean potential surrounding quotes or punctuation
        label = label.replace('"', '').replace("'", "").strip()
        return {"label": str(label), "confidence": 0.95}
    except Exception as e:
        print(f"GPT few-shot inference failed: {e}")
        return {"label": "unavailable", "confidence": 0.0}


def predict_gpt_zeroshot(text):
    if openai_client is None:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        messages = [
            {"role": "system", "content": "You are an IT support ticket classifier. Classify into exactly one of: Fileservice, Support general, Software, O365, Active Directory, Computer-Services. Reply with the class name only."},
            {"role": "user", "content": text}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        label = response.choices[0].message.content.strip()
        label = label.replace('"', '').replace("'", "").strip()
        return {"label": str(label), "confidence": 0.95}
    except Exception as e:
        print(f"GPT zero-shot inference failed: {e}")
        return {"label": "unavailable", "confidence": 0.0}


def predict_lora(text):
    if lora_model is None or lora_tokenizer is None:
        return {"label": "unavailable", "confidence": 0.0}
    try:
        inputs = lora_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        # Run on CPU or GPU depending on hardware
        device = next(lora_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = lora_model(**inputs)
            
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = float(probs[0][pred_id].item())
        label = id2label.get(pred_id, "unavailable")
        return {"label": str(label), "confidence": confidence}
    except Exception as e:
        print(f"LoRA inference failed: {e}")
        return {"label": "unavailable", "confidence": 0.0}


# ==========================================
# ROUTE ENDPOINTS
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing ticket text input"}), 400
        
    text = data["text"]
    
    tfidf_res = predict_tfidf(text)
    embeddings_res = predict_embeddings(text)
    gpt_res = predict_gpt(text)
    lora_res = predict_lora(text)
    embeddings_augmented_res = predict_embeddings_augmented(text)
    gpt_zeroshot_res = predict_gpt_zeroshot(text)
    
    return jsonify({
        "tfidf": tfidf_res,
        "embeddings": embeddings_res,
        "gpt_fewshot": gpt_res,
        "lora": lora_res,
        "embeddings_augmented": embeddings_augmented_res,
        "gpt_zeroshot": gpt_zeroshot_res
    })


@app.route("/static/images/<path:filename>")
def serve_static_images(filename):
    images_dir = os.path.join(STATIC_DIR, "images")
    return send_from_directory(images_dir, filename)


if __name__ == "__main__":
    print("Starting IT Support Ticket Classifier Flask Server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
