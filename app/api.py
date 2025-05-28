from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Tuple  # Added Tuple here
import pickle
import re
import numpy as np
import logging
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Configure logging with a detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Legal Document Classifier API",
    version="1.1.0",
    description="API for classifying legal case reports into areas of law using TF-IDF or BERT models.",
    docs_url="/docs",
    openapi_tags=[
        {"name": "Prediction", "description": "Endpoints for predicting areas of law."},
        {"name": "Health", "description": "Endpoint to check API health."},
        {"name": "Analytics", "description": "Endpoint for usage statistics."},
        {"name": "Root", "description": "Root endpoint with API information."}
    ]
)

# Global variables for models and label mapping
class ModelConfig:
    """Static class to hold model configurations and mappings."""
    LABEL_MAP: Dict[int, str] = {
        0: "Civil Procedure",
        1: "Enforcement of Fundamental Rights",
        2: "Election Petition",
        3: "Other",
        4: "Criminal Law",
        5: "Property Law"
    }
    REVERSE_LABEL_MAP: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}

# Load models and vectorizer at startup
try:
    logger.info("Loading TF-IDF model and vectorizer...")
    with open('../models/saved_model.pkl', 'rb') as f:
        TFIDF_MODEL = pickle.load(f)
    with open('../models/vectorizer.pkl', 'rb') as f:
        VECTORIZER = pickle.load(f)

    logger.info("Loading BERT model and tokenizer...")
    BERT_MODEL = BertForSequenceClassification.from_pretrained('../models/bert_model')
    TOKENIZER = BertTokenizer.from_pretrained('../models/bert_tokenizer')
except FileNotFoundError as e:
    logger.error(f"Failed to load model files: {e}")
    raise HTTPException(status_code=500, detail="Model or vectorizer files not found.")
except Exception as e:
    logger.error(f"Unexpected error while loading models: {e}")
    raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

# Utility functions
def clean_text(text: str) -> str:
    """
    Clean a legal text by removing HTML entities, citations, special characters, and normalizing spaces.
    
    Args:
        text (str): Raw input text to clean.
    
    Returns:
        str: Cleaned text, or empty string if input is invalid.
    """
    if not isinstance(text, str):
        logger.warning("Input text is not a string, returning empty string.")
        return ""
    
    text = text.lower()
    text = re.sub(r'&\w+;', ' ', text)  # Remove HTML entities
    text = re.sub(r'\[\d{4}\].*?\(pt\.\s*\d+\)', ' ', text)  # Remove citations
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# Prediction functions
def predict_with_tfidf(text: str) -> Tuple[int, float]:
    """
    Predict the area of law using the TF-IDF model.
    
    Args:
        text (str): Cleaned text to classify.
    
    Returns:
        Tuple[int, float]: Predicted label index and confidence score.
    
    Raises:
        ValueError: If the cleaned text is empty.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Input text is empty after cleaning.")
    
    text_vector = VECTORIZER.transform([cleaned_text])
    prediction = TFIDF_MODEL.predict(text_vector)[0]
    confidence = float(TFIDF_MODEL.predict_proba(text_vector)[0].max())
    return prediction, confidence

def predict_with_bert(text: str) -> Tuple[int, float]:
    """
    Predict the area of law using the BERT model.
    
    Args:
        text (str): Cleaned text to classify.
    
    Returns:
        Tuple[int, float]: Predicted label index and confidence score.
    
    Raises:
        ValueError: If the cleaned text is empty.
    """
    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise ValueError("Input text is empty after cleaning.")
    
    inputs = TOKENIZER(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = BERT_MODEL(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = float(torch.softmax(outputs.logits, dim=1)[0].max().item())
    return prediction, confidence

# Request models
class CaseReport(BaseModel):
    """Model for a single legal case report input."""
    full_report: str

    @validator('full_report')
    def validate_length(cls, value: str) -> str:
        if len(value.strip()) < 50:
            raise ValueError("Full report must be at least 50 characters long.")
        return value

class BatchCaseReport(BaseModel):
    """Model for batch input of legal case reports."""
    full_reports: List[str]
    model_type: Optional[str] = "tfidf"  # Default to TF-IDF model

    # Resolve Pydantic namespace conflict
    model_config = {"protected_namespaces": ()}

    @validator('full_reports')
    def validate_batch_size(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Batch cannot be empty.")
        if len(value) > 100:
            raise ValueError("Batch size cannot exceed 100 reports.")
        return [report for report in value if len(report.strip()) >= 50]

    @validator('model_type')
    def validate_model_type(cls, value: str) -> str:
        if value not in ["tfidf", "bert"]:
            raise ValueError("Model type must be 'tfidf' or 'bert'.")
        return value

# API Endpoints
@app.get("/", response_model=None, tags=["Root"])
async def root_endpoint() -> Dict[str, str]:
    """Root endpoint providing a welcome message and API navigation."""
    return {
        "message": "Welcome to the Legal Document Classifier API!",
        "documentation": "/docs",
        "available_endpoints": ["/predict", "/predict-batch", "/health", "/stats"]
    }

@app.get("/health", response_model=Dict[str, str], tags=["Health"])
async def check_health() -> Dict[str, str]:
    """Check the health status of the API."""
    return {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/stats", response_model=Dict[str, str], tags=["Analytics"])
async def usage_statistics() -> Dict[str, str]:
    """
    Retrieve usage statistics for the API.
    
    Note: Currently returns simulated data; replace with actual tracking in production.
    """
    return {
        "total_requests": "100",  # Placeholder; implement counter in production
        "average_confidence": "0.85",  # Placeholder; compute from logged predictions
        "last_updated": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=Dict[str, str], tags=["Prediction"])
async def classify_single_report(case: CaseReport, model_type: Optional[str] = "tfidf") -> Dict[str, str]:
    """
    Classify a single legal case report into an area of law.
    
    Args:
        case (CaseReport): Input containing the full legal report.
        model_type (str, optional): Model to use ('tfidf' or 'bert'). Defaults to 'tfidf'.
    
    Returns:
        Dict: Prediction result with area of law, confidence, model used, and input length.
    
    Raises:
        HTTPException: For invalid inputs (400) or server errors (500).
    """
    try:
        if model_type not in ["tfidf", "bert"]:
            raise ValueError("Model type must be 'tfidf' or 'bert'.")

        prediction_func = predict_with_tfidf if model_type == "tfidf" else predict_with_bert
        label_idx, confidence = prediction_func(case.full_report)
        area_of_law = ModelConfig.LABEL_MAP.get(label_idx, "Unknown")

        logger.info(f"Prediction: {area_of_law} (Confidence: {confidence:.2f}) using {model_type}")
        return {
            "area_of_law": area_of_law,
            "confidence": str(confidence),
            "model_used": model_type,
            "input_length": str(len(case.full_report))
        }
    except ValueError as ve:
        logger.warning(f"Invalid input: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=List[Dict[str, str]], tags=["Prediction"])
async def classify_batch_reports(batch: BatchCaseReport) -> List[Dict[str, str]]:
    """
    Classify multiple legal case reports into areas of law.
    
    Args:
        batch (BatchCaseReport): Batch input containing a list of reports and model type.
    
    Returns:
        List[Dict]: List of prediction results for each report.
    
    Raises:
        HTTPException: For invalid inputs (400) or server errors (500).
    """
    try:
        cleaned_texts = [clean_text(report) for report in batch.full_reports]
        if not any(cleaned_texts):
            raise ValueError("All reports are empty after cleaning.")

        results = []
        if batch.model_type == "tfidf":
            text_vectors = VECTORIZER.transform(cleaned_texts)
            predictions = TFIDF_MODEL.predict(text_vectors)
            probabilities = TFIDF_MODEL.predict_proba(text_vectors)
            results = [
                {
                    "area_of_law": ModelConfig.LABEL_MAP.get(pred, "Unknown"),
                    "confidence": str(float(probs.max())),
                    "model_used": "tfidf"
                }
                for pred, probs in zip(predictions, probabilities)
            ]
        else:  # BERT
            for text in cleaned_texts:
                if not text:
                    continue
                label_idx, confidence = predict_with_bert(text)
                results.append({
                    "area_of_law": ModelConfig.LABEL_MAP.get(label_idx, "Unknown"),
                    "confidence": str(confidence),
                    "model_used": "bert"
                })

        logger.info(f"Processed batch of {len(batch.full_reports)} reports using {batch.model_type}")
        return results
    except ValueError as ve:
        logger.warning(f"Invalid batch input: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")