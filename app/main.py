"""
FastAPI Backend for NLP IDU Project
====================================
Provides REST API endpoints for document analysis using the NLP IDU pipeline.

Start with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import logging
from datetime import datetime
import joblib
import spacy
import os

from src.logger import get_logger
from src.pipeline import NLPInferencePipeline, PredictionPipeline
from src.components import TextSummarizer
from src.exceptions import NLPException

# ============================================================================
# LOGGING
# ============================================================================

logger = get_logger(__name__)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NLP IDU API",
    description="Intelligent Document Understanding - NLP Pipeline API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS CONFIGURATION
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (modify for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AnalysisRequest(BaseModel):
    """Request model for document analysis."""
    
    text: str = Field(
        ...,
        min_length=20,
        max_length=50000,
        description="Document text to analyze"
    )
    
    task: Literal[
        "full_pipeline",
        "classification",
        "named_entity_recognition",
        "summarization"
    ] = Field(
        default="full_pipeline",
        description="Type of analysis to perform"
    )


class Entity(BaseModel):
    """Model for named entity."""
    text: str
    label: str
    start: Optional[int] = None
    end: Optional[int] = None


class EntitySummary(BaseModel):
    """Summary of entities by type."""
    entity_type: str
    count: int
    examples: List[str]


class AnalysisResponse(BaseModel):
    """Response model for document analysis."""
    
    success: bool = True
    message: str = "Analysis completed successfully"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Analysis Results
    data: Dict = Field(
        default_factory=dict,
        description="Analysis results"
    )
    
    # Individual fields for easier access
    category: Optional[str] = None
    entities: Optional[List[Entity]] = None
    entity_summary: Optional[Dict[str, int]] = None
    summary: Optional[str] = None
    
    # Metadata
    text_length: int = 0
    word_count: int = 0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    models_loaded: bool = False


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# GLOBAL STATE - MODEL INITIALIZATION
# ============================================================================

class ModelManager:
    """Manages model loading and lifecycle."""
    
    def __init__(self):
        self.inference_pipeline = None
        self.prediction_pipeline = None
        self.is_loaded = False
        self.load_error = None
    
    def load_models(self):
        """Load all required models."""
        try:
            logger.info("Loading NLP IDU models...")
            
            # Try to load inference pipeline
            try:
                self.inference_pipeline = NLPInferencePipeline(
                    classifier_path="models/text_classifier.pkl",
                    vectorizer_path="models/tfidf_vectorizer.pkl",
                    ner_model="en_core_web_sm",
                    summarizer_model="t5-small"
                )
                logger.info("✓ Inference pipeline loaded")
            except FileNotFoundError as e:
                logger.warning(f"Inference pipeline models not found: {e}")
                logger.info("Will use fallback models...")
                
                # Fallback: load individual components
                try:
                    nlp = spacy.load("en_core_web_sm")
                    summarizer = TextSummarizer("t5-small")
                    
                    self.prediction_pipeline = PredictionPipeline(
                        classifier_model=None,
                        vectorizer=None,
                        nlp_model=nlp,
                        summarizer=summarizer
                    )
                    logger.info("✓ Fallback models loaded (NER + Summarization)")
                except Exception as fallback_error:
                    logger.error(f"Fallback loading failed: {fallback_error}")
                    self.load_error = str(fallback_error)
                    return False
            
            self.is_loaded = True
            logger.info("✓ All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.load_error = str(e)
            return False


# Global model manager instance
model_manager = ModelManager()


# ============================================================================
# STARTUP & SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("🚀 FastAPI server starting...")
    model_manager.load_models()
    if model_manager.is_loaded:
        logger.info("✅ Server ready for requests")
    else:
        logger.warning("⚠️ Server started but models failed to load")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 FastAPI server shutting down...")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text_statistics(text: str) -> Dict:
    """Calculate text statistics."""
    words = text.split()
    sentences = [s for s in text.split(".") if s.strip()]
    
    return {
        "text_length": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": round(len(text) / len(words), 2) if words else 0
    }


def format_entities(entities: List[Dict]) -> tuple:
    """Format entities for response.
    
    Returns:
        Tuple of (entities_list, entity_summary_dict)
    """
    if not entities:
        return [], {}
    
    # Convert to Entity objects if needed
    formatted_entities = []
    entity_summary = {}
    
    for ent in entities:
        if isinstance(ent, dict):
            formatted_entities.append(Entity(**ent))
            
            # Count by label
            label = ent.get("label", "UNKNOWN")
            entity_summary[label] = entity_summary.get(label, 0) + 1
        else:
            # Handle tuple format (text, label)
            if isinstance(ent, (tuple, list)) and len(ent) >= 2:
                formatted_entities.append(Entity(text=ent[0], label=ent[1]))
                entity_summary[ent[1]] = entity_summary.get(ent[1], 0) + 1
    
    return formatted_entities, entity_summary


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="operational" if model_manager.is_loaded else "degraded",
        models_loaded=model_manager.is_loaded
    )


# ============================================================================
# MAIN ANALYSIS ENDPOINT
# ============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(request: AnalysisRequest):
    """
    Analyze document using NLP IDU pipeline.
    
    Parameters:
    -----------
    - text: Document text to analyze
    - task: Type of analysis (full_pipeline, classification, ner, summarization)
    
    Returns:
    --------
    - Analysis results including classification, entities, and summary
    """
    try:
        logger.info(f"Analyzing document (task={request.task}, len={len(request.text)})")
        
        # Validate input
        if not request.text or len(request.text.strip()) < 20:
            raise ValueError("Text must be at least 20 characters")
        
        # Get text statistics
        stats = get_text_statistics(request.text)
        
        # Initialize response
        response_data = {
            "category": None,
            "entities": [],
            "entity_summary": {},
            "summary": None
        }
        
        # =====================================================================
        # FULL PIPELINE
        # =====================================================================
        if request.task == "full_pipeline":
            
            if model_manager.inference_pipeline:
                logger.info("Using inference pipeline for full analysis")
                
                result = model_manager.inference_pipeline.process_document(
                    request.text
                )
                
                response_data["category"] = result.get("category")
                response_data["entities"] = result.get("entities", [])
                response_data["summary"] = result.get("summary")
                
            else:
                logger.warning("Inference pipeline not available, using fallback")
                
                # Fallback: Do NER + Summarization
                if model_manager.prediction_pipeline:
                    result = model_manager.prediction_pipeline.predict(
                        request.text,
                        tasks=["ner", "summarization"]
                    )
                    
                    response_data["entities"] = result.get("entities", [])
                    response_data["summary"] = result.get("summary")
                else:
                    raise NLPException("No models available for analysis")
        
        # =====================================================================
        # CLASSIFICATION ONLY
        # =====================================================================
        elif request.task == "classification":
            
            if model_manager.inference_pipeline:
                result = model_manager.inference_pipeline.process_document(
                    request.text
                )
                response_data["category"] = result.get("category")
            else:
                logger.warning("Classification model not available")
                response_data["category"] = "Unknown (Model not loaded)"
        
        # =====================================================================
        # NER ONLY
        # =====================================================================
        elif request.task == "named_entity_recognition":
            
            if model_manager.prediction_pipeline:
                result = model_manager.prediction_pipeline.predict(
                    request.text,
                    tasks=["ner"]
                )
                response_data["entities"] = result.get("entities", [])
            else:
                raise NLPException("NER model not loaded")
        
        # =====================================================================
        # SUMMARIZATION ONLY
        # =====================================================================
        elif request.task == "summarization":
            
            if model_manager.prediction_pipeline:
                result = model_manager.prediction_pipeline.predict(
                    request.text,
                    tasks=["summarization"]
                )
                response_data["summary"] = result.get("summary")
            else:
                raise NLPException("Summarizer model not loaded")
        
        # =====================================================================
        # FORMAT RESPONSE
        # =====================================================================
        
        # Format entities
        entities, entity_summary = format_entities(
            response_data.get("entities", [])
        )
        
        response_data["entities"] = [e.dict() for e in entities]
        response_data["entity_summary"] = entity_summary
        
        # Build response
        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            data=response_data,
            category=response_data.get("category"),
            entities=entities,
            entity_summary=entity_summary,
            summary=response_data.get("summary"),
            text_length=stats["text_length"],
            word_count=stats["word_count"]
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except NLPException as e:
        logger.error(f"NLP error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# BATCH ANALYSIS ENDPOINT
# ============================================================================

@app.post("/analyze-batch")
async def analyze_batch(
    documents: List[Dict[str, str]],
    task: str = "full_pipeline"
):
    """
    Analyze multiple documents in batch.
    
    Parameters:
    -----------
    - documents: List of dicts with 'text' and optional 'id'
    - task: Type of analysis
    
    Returns:
    --------
    - List of analysis results
    """
    try:
        if not documents or len(documents) > 100:
            raise ValueError("Provide 1-100 documents")
        
        results = []
        
        for doc in documents:
            request = AnalysisRequest(text=doc.get("text", ""), task=task)
            result = await analyze_document(request)
            
            result_dict = result.dict()
            if "id" in doc:
                result_dict["id"] = doc["id"]
            
            results.append(result_dict)
        
        return {
            "success": True,
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# MODEL STATUS ENDPOINT
# ============================================================================

@app.get("/models/status")
async def model_status():
    """Get status of loaded models."""
    return {
        "inference_pipeline_loaded": model_manager.inference_pipeline is not None,
        "prediction_pipeline_loaded": model_manager.prediction_pipeline is not None,
        "models_ready": model_manager.is_loaded,
        "load_error": model_manager.load_error,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "NLP IDU API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /analyze": "Analyze single document",
            "POST /analyze-batch": "Analyze multiple documents",
            "GET /health": "Health check",
            "GET /models/status": "Model status"
        }
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# IF RUN DIRECTLY
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Pull dynamic web port allocated dynamically by Render's environment router.
    # Defaults safely back to local loop container '8000'.
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False if os.environ.get("PORT") else True
    )