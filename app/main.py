from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import spacy
from transformers import pipeline

# ---------------------------
# App init
# ---------------------------
app = FastAPI(
    title="Intelligent Document Understanding API",
    description="Document Classification, NER, and Summarization",
    version="1.0"
)

# ---------------------------
# Load models ONCE at startup
# ---------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

clf = joblib.load(MODEL_DIR / "text_classifier.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")


nlp = spacy.load("en_core_web_sm")

summarizer = pipeline(
    "text2text-generation",
    model="t5-small",
    device=-1
)

# ---------------------------
# Request schema
# ---------------------------
class TextRequest(BaseModel):
    text: str

# ---------------------------
# Health check
# ---------------------------
@app.get("/")
def health():
    return {"status": "IDU API running"}

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict(request: TextRequest):
    text = request.text

    # 1. Classification
    X = vectorizer.transform([text])
    category = clf.predict(X)[0]

    # 2. NER
    doc_nlp = nlp(text)

    VALID_LABELS = {"ORG", "GPE", "PERSON", "MONEY", "DATE"}
    entities = [
        (ent.text, ent.label_)
        for ent in doc_nlp.ents
        if ent.label_ in VALID_LABELS
    ]

    # 3. Summarization
    summary = summarizer(
        "summarize: " + text,   
        max_new_tokens=120,    
        do_sample=False
    )[0]["generated_text"] 

    return {
        "category": category,
        "entities": entities,
        "summary": summary
    }
