"""
FastAPI Integration
===================

Production-ready FastAPI server for NLP pipeline.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import InferencePipeline
from src.pipeline.inference_pipeline import PipelineConfig, TaskType
from src.logger import get_logger, setup_logging
from src.exceptions import NLPPipelineException

# Setup logging
setup_logging("nlp_pipeline_api")
logger = get_logger("nlp_pipeline_api")

# Initialize FastAPI app
app = FastAPI(
    title="NLP Pipeline API",
    description="Production NLP pipeline with classification, NER, and summarization",
    version="1.0.0"
)

# Initialize pipeline with error handling
try:
    logger.info("Initializing InferencePipeline...")
    pipeline = InferencePipeline(
        config=PipelineConfig(
            # Update with your actual model paths
            classifier_model_path=None,
            classifier_vectorizer_path=None,
            ner_model_name="en_core_web_sm",
            summarizer_model_name="t5-small"
        )
    )
    logger.info("InferencePipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
    logger.warning("Pipeline will be unavailable. Check model paths and dependencies.")
    pipeline = None


# Pydantic models for request/response
class TaskEnum(str, Enum):
    """Task types."""
    CLASSIFICATION = "classification"
    NER = "named_entity_recognition"
    SUMMARIZATION = "summarization"
    FULL = "full_pipeline"


class TextInput(BaseModel):
    """Input model for text processing."""
    text: str = Field(..., min_length=20, max_length=10000, description="Input text")
    task: TaskEnum = Field(default=TaskEnum.FULL, description="Task type")


class ClassificationRequest(BaseModel):
    """Classification request model."""
    text: str = Field(..., min_length=20, description="Text to classify")


class SummarizationRequest(BaseModel):
    """Summarization request model."""
    text: str = Field(..., min_length=20, description="Text to summarize")
    max_length: int = Field(default=130, ge=30, le=300, description="Max summary length")


class NERRequest(BaseModel):
    """NER request model."""
    text: str = Field(..., min_length=20, description="Text for NER")


class BatchProcessRequest(BaseModel):
    """Batch processing request."""
    texts: List[str] = Field(..., description="List of texts to process")
    task: TaskEnum = Field(default=TaskEnum.FULL, description="Task type")


class Response(BaseModel):
    """Standard response model."""
    status: str
    data: dict
    message: Optional[str] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Check pipeline health status."""
    if pipeline is None:
        logger.warning("Health check requested but pipeline not initialized")
        return {
            "status": "degraded",
            "message": "Pipeline not fully initialized",
            "components": {
                "classifier_ready": False,
                "ner_ready": False,
                "summarizer_ready": False
            }
        }
    
    try:
        status = pipeline.get_system_status()
        return {
            "status": "healthy",
            "components": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Classification endpoint
@app.post("/classify", response_model=Response)
async def classify(request: ClassificationRequest):
    """
    Classify input text.
    
    - **text**: Input text (min 20 chars)
    - Returns: Predicted category
    """
    if pipeline is None:
        logger.error("Classification request but pipeline not initialized")
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Check logs for initialization errors."
        )
    
    try:
        logger.debug(f"Classification request received")
        result = pipeline.classify(request.text)
        logger.info("Classification completed successfully")
        return Response(
            status="success",
            data=result,
            message="Classification completed"
        )
    except NLPPipelineException as e:
        logger.warning(f"Classification validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# NER endpoint
@app.post("/extract-entities", response_model=Response)
async def extract_entities(request: NERRequest):
    """
    Extract named entities from text.
    
    - **text**: Input text (min 20 chars)
    - Returns: List of entities with labels
    """
    if pipeline is None:
        logger.error("NER request but pipeline not initialized")
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Check logs for initialization errors."
        )
    
    try:
        logger.debug(f"NER request received")
        result = pipeline.extract_entities(request.text)
        logger.info("Entity extraction completed successfully")
        return Response(
            status="success",
            data=result,
            message="Entity extraction completed"
        )
    except NLPPipelineException as e:
        logger.warning(f"NER validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"NER error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Summarization endpoint
@app.post("/summarize", response_model=Response)
async def summarize(request: SummarizationRequest):
    """
    Generate text summary.
    
    - **text**: Input text (min 20 chars)
    - **max_length**: Maximum summary length (30-300)
    - Returns: Generated summary
    """
    if pipeline is None:
        logger.error("Summarization request but pipeline not initialized")
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Check logs for initialization errors."
        )
    
    try:
        logger.debug(f"Summarization request received")
        result = pipeline.summarize(request.text, max_length=request.max_length)
        logger.info("Summarization completed successfully")
        return Response(
            status="success",
            data=result,
            message="Summarization completed"
        )
    except NLPPipelineException as e:
        logger.warning(f"Summarization validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Full analysis endpoint
@app.post("/analyze", response_model=Response)
async def full_analysis(request: TextInput):
    """
    Perform complete document analysis.
    
    - **text**: Input text
    - **task**: Task type (classification, ner, summarization, full_pipeline)
    - Returns: Results from specified task(s)
    """
    if pipeline is None:
        logger.error("Analysis request but pipeline not initialized")
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Check logs for initialization errors."
        )
    
    try:
        logger.debug(f"Analysis request received for task: {request.task.value}")
        
        # Map string to TaskType enum
        task_map = {
            "classification": TaskType.CLASSIFICATION,
            "named_entity_recognition": TaskType.NER,
            "summarization": TaskType.SUMMARIZATION,
            "full_pipeline": TaskType.FULL
        }
        
        task_type = task_map.get(request.task.value, TaskType.FULL)
        result = pipeline.process(request.text, task=task_type)
        
        logger.info(f"Analysis completed for task: {request.task.value}")
        
        return Response(
            status="success",
            data=result,
            message=f"Analysis completed with task: {request.task.value}"
        )
    except NLPPipelineException as e:
        logger.warning(f"Analysis validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Batch processing endpoint
@app.post("/batch-analyze", response_model=dict)
async def batch_analyze(request: BatchProcessRequest):
    """
    Process multiple texts.
    
    - **texts**: List of texts to process
    - **task**: Task type
    - Returns: Results for each text
    """
    if pipeline is None:
        logger.error("Batch processing request but pipeline not initialized")
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Check logs for initialization errors."
        )
    
    try:
        logger.debug(f"Batch processing request received for {len(request.texts)} texts")
        
        task_map = {
            "classification": TaskType.CLASSIFICATION,
            "named_entity_recognition": TaskType.NER,
            "summarization": TaskType.SUMMARIZATION,
            "full_pipeline": TaskType.FULL
        }
        
        task_type = task_map.get(request.task.value, TaskType.FULL)
        results = pipeline.batch_process(request.texts, task=task_type)
        
        logger.info(f"Batch processing completed for {len(results)} texts")
        
        return {
            "status": "success",
            "count": len(results),
            "results": results,
            "message": f"Batch processing completed for {len(results)} texts"
        }
    except NLPPipelineException as e:
        logger.warning(f"Batch processing validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "NLP Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "classify": "/classify",
            "extract_entities": "/extract-entities",
            "summarize": "/summarize",
            "full_analysis": "/analyze",
            "batch_processing": "/batch-analyze",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python -m uvicorn app.main:app --reload
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )