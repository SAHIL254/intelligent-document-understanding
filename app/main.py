"""
FastAPI Backend for NLP IDU Project
====================================
Production-ready FastAPI backend for Intelligent Document Understanding.
Optimized with Lazy Weight Extraction and Dynamic Segment Isolation.

Render Start Command:
uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from contextlib import asynccontextmanager
from datetime import datetime
import os
import spacy
import gc  # Aggressive garbage collection context

# ============================================================================
# MEMORY & PROCESS OPTIMIZATIONS (CRITICAL FOR RENDER FREE TIER)
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from src.logger import get_logger
from src.pipeline import NLPInferencePipeline, PredictionPipeline
from src.components import TextSummarizer
from src.exceptions import NLPException

logger = get_logger(__name__)

# ============================================================================
# MODEL MANAGER (UPDATED FOR LAZY INITIALIZATION)
# ============================================================================
class ModelManager:
    """Handles model loading and inference pipelines dynamically to avoid OOM errors."""

    def __init__(self):
        self.inference_pipeline = None
        self.prediction_pipeline = None
        self.is_loaded = False
        self.load_error = None
        self.has_local_artifacts = False  # Track state globally instead of hitting disk on requests

    def initialize_base_environment(self):
        """Prepares metadata and tests path structures without loading heavy weights."""
        try:
            logger.info("Initializing base application environment configurations...")
            
            # Cache the disk check right here on server startup
            if os.path.exists("models/text_classifier.pkl") and os.path.exists("models/tfidf_vectorizer.pkl"):
                logger.info("✓ Main pipeline artifacts verified on disk.")
                self.has_local_artifacts = True
            else:
                logger.warning("⚠️ Main pipeline models missing. Fallback pipeline will load lazily on request.")
                self.has_local_artifacts = False
            
            # Flag true because the API layer is open and ready to map connections
            self.is_loaded = True
            
        except Exception as e:
            self.load_error = str(e)
            self.is_loaded = False
            logger.error(f"Environment initialization failed: {e}")

    def get_inference_pipeline(self):
        """Lazy loader for the main complex inference pipeline."""
        if self.inference_pipeline is None:
            logger.info("📥 Lazy loading full inference pipeline into RAM...")
            self.inference_pipeline = NLPInferencePipeline(
                classifier_path="models/text_classifier.pkl",
                vectorizer_path="models/tfidf_vectorizer.pkl",
                ner_model="en_core_web_sm",
                summarizer_model="t5-small"
            )
            gc.collect()  # Clean up temporary allocation fragments
        return self.inference_pipeline

    def get_prediction_pipeline(self):
        """Lazy loader for the prediction pipeline weights."""
        if self.prediction_pipeline is None:
            logger.info("📥 Lazy loading fallback pipeline components (spaCy + T5-small)...")
            
            nlp = spacy.load("en_core_web_sm")
            summarizer = TextSummarizer("t5-small")

            self.prediction_pipeline = PredictionPipeline(
                classifier_model=None,
                vectorizer=None,
                nlp_model=nlp,
                summarizer=summarizer
            )
            gc.collect()
        return self.prediction_pipeline

# Global model manager instance
model_manager = ModelManager()

# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting NLP IDU API...")
    
    # Do NOT load full weights here anymore! Initialize environment instead.
    model_manager.initialize_base_environment()

    if model_manager.is_loaded:
        logger.info("✅ API port ready to open instantly.")
    else:
        logger.warning("⚠️ API started with initial configuration issues")

    yield

    logger.info("🛑 Shutting down NLP IDU API...")

# ============================================================================
# FASTAPI APP SETUP & VALIDATIONS
# ============================================================================
app = FastAPI(
    title="NLP IDU API",
    description="Intelligent Document Understanding API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=20, max_length=50000)
    task: Literal["full_pipeline", "classification", "named_entity_recognition", "summarization"] = "full_pipeline"

class Entity(BaseModel):
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None

class AnalysisResponse(BaseModel):
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    category: Optional[str] = None
    entities: Optional[List[Entity]] = None
    entity_summary: Optional[Dict[str, int]] = None
    summary: Optional[str] = None
    text_length: int = 0
    word_count: int = 0

def get_text_statistics(text: str):
    words = text.split()
    return {"text_length": len(text), "word_count": len(words)}

def format_entities(entities):
    formatted_entities = []
    entity_summary = {}
    if not entities:
        return formatted_entities, entity_summary
    for ent in entities:
        if isinstance(ent, dict):
            formatted_entities.append(Entity(**ent))
            label = ent.get("label", "UNKNOWN")
            entity_summary[label] = entity_summary.get(label, 0) + 1
    return formatted_entities, entity_summary

@app.get("/")
async def root():
    return {"name": "NLP IDU API", "status": "running", "version": "1.0.0", "docs": "/docs"}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model_manager.is_loaded else "degraded",
        "models_initialized": model_manager.is_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/status")
async def model_status():
    return {
        "inference_pipeline_allocated": model_manager.inference_pipeline is not None,
        "prediction_pipeline_allocated": model_manager.prediction_pipeline is not None,
        "environment_ready": model_manager.is_loaded
    }

# ============================================================================
# ANALYZE DOCUMENT ENDPOINT (WITH DYNAMIC PIPELINE RETRIEVAL)
# ============================================================================
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(request: AnalysisRequest):
    try:
        logger.info(f"Analyzing document | task={request.task}")
        stats = get_text_statistics(request.text)
        response_data = {"category": None, "entities": [], "summary": None}

        # ================================================================
        # FULL PIPELINE EXECUTION
        # ================================================================
        if request.task == "full_pipeline":
            if model_manager.has_local_artifacts:
                pipeline = model_manager.get_inference_pipeline()
                result = pipeline.process_document(request.text)
                response_data["category"] = result.get("category")
                response_data["entities"] = result.get("entities", [])
                response_data["summary"] = result.get("summary")
            else:
                pipeline = model_manager.get_prediction_pipeline()
                result = pipeline.predict(request.text, tasks=["ner", "summarization"])
                response_data["entities"] = result.get("entities", [])
                response_data["summary"] = result.get("summary")

        # ================================================================
        # SINGLE CLASSIFICATION TASK
        # ================================================================
        elif request.task == "classification":
            if not model_manager.has_local_artifacts:
                raise NLPException("Classification model artifacts unavailable on disk")
            pipeline = model_manager.get_inference_pipeline()
            result = pipeline.process_document(request.text)
            response_data["category"] = result.get("category")

        # ================================================================
        # SINGLE NER TASK
        # ================================================================
        elif request.task == "named_entity_recognition":
            pipeline = model_manager.get_prediction_pipeline()
            result = pipeline.predict(request.text, tasks=["ner"])
            response_data["entities"] = result.get("entities", [])

        # ================================================================
        # SINGLE SUMMARIZATION TASK
        # ================================================================
        elif request.task == "summarization":
            pipeline = model_manager.get_prediction_pipeline()
            result = pipeline.predict(request.text, tasks=["summarization"])
            response_data["summary"] = result.get("summary")

        # ================================================================
        # FORMAT AND RETURN RESPONSE
        # ================================================================
        entities, entity_summary = format_entities(response_data["entities"])
        
        # Explicit garbage collection call post-inference to clean memory leaks
        gc.collect()

        return AnalysisResponse(
            success=True,
            category=response_data["category"],
            entities=entities,
            entity_summary=entity_summary,
            summary=response_data["summary"],
            text_length=stats["text_length"],
            word_count=stats["word_count"]
        )

    except NLPException as e:
        logger.error(f"NLP Error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Internal Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)