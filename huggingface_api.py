import os
import requests
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env (for HF API key)
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L12-v2"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

def get_embeddings(texts):
    """
    Generate sentence embeddings using Hugging Face API.
    Falls back gracefully if the API fails.
    """
    if not texts or not isinstance(texts, list):
        raise ValueError("get_embeddings() expects a list of texts")

    payload = {"inputs": texts}
    try:
        print("⚙️ Requesting embeddings from Hugging Face API...")

        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        # The API sometimes returns nested structures
        embeddings = []
        for emb in data:
            if isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb):
                embeddings.append(emb)
            elif isinstance(emb, dict) and "embedding" in emb:
                embeddings.append(emb["embedding"])
            else:
                # Unrecognized format
                raise ValueError(f"Unexpected embedding format: {type(emb)}")

        print(f"✅ Received {len(embeddings)} embeddings from HF API.")
        return np.array(embeddings, dtype=float)

    except Exception as e:
        print(f"⚠️ Hugging Face API failed: {e}")
        print("🔄 Falling back to local model (SentenceTransformer).")
        return _local_embeddings(texts)

def _local_embeddings(texts):
    """
    Fallback using a local sentence-transformer model
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("🧠 Loading local model for embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
        print("✅ Local embeddings generated successfully.")
        return np.array(embeddings, dtype=float)
    except Exception as e:
        print(f"❌ Local model embedding generation failed: {e}")
        raise RuntimeError("Both Hugging Face and local embedding generation failed.")

def analyze_sentiment(text):
    """
    Analyze sentiment of a text using Hugging Face sentiment model.
    Returns: {"label": "POSITIVE" or "NEGATIVE" or "NEUTRAL", "score": float}
    """
    if not text or not isinstance(text, str):
        return {"label": "NEUTRAL", "score": 0.0}

    SENTIMENT_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"

    try:
        response = requests.post(SENTIMENT_URL, headers=HEADERS, json={"inputs": text}, timeout=30)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            result = result[0]  # flatten [[...]] to [...]
        if isinstance(result, list):
            best = max(result, key=lambda x: x.get("score", 0))
            return {"label": best.get("label", "NEUTRAL"), "score": float(best.get("score", 0))}

        return {"label": "NEUTRAL", "score": 0.0}

    except Exception as e:
        print(f"⚠️ Sentiment API failed: {e}")
        return {"label": "NEUTRAL", "score": 0.0}

