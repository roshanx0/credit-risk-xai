"""
FastAPI Backend for Credit Risk Assessment System
Provides prediction and explanation endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from risk_engine import CreditRiskEngine
from explainer import CreditRiskExplainer
from utils import ensure_serializable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Assessment API",
    description="Interpretable credit default prediction with XGBoost and SHAP",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
model = None
preprocessor = None
explainer = None
risk_engine = None
feature_names = None


def apply_hard_business_rules(probability: float, application, foir: float) -> float:
    """
    Apply hard business rules that override ML predictions.
    These are non-negotiable red flags in Indian banking based on regulatory requirements
    and industry best practices.
    
    Args:
        probability: ML model's predicted default probability
        application: LoanApplication object with applicant details
        foir: Fixed Obligation to Income Ratio (%)
    
    Returns: Adjusted probability (max of ML prediction and rule-based minimum)
    """
    original_prob = probability
    
    # Rule 1: DPD (Days Past Due) - Payment history is factual, not predictive
    if application.dpd_last_12m > 90:  # 90+ days = NPA (Non-Performing Asset)
        probability = max(probability, 0.93)
    elif application.dpd_last_12m > 60:  # 60-90 days = about to be NPA
        probability = max(probability, 0.85)
    elif application.dpd_last_12m > 30:  # 30-60 days = serious delinquency
        probability = max(probability, 0.72)
    
    # Rule 2: CIBIL Score - Industry standard thresholds
    if application.cibil_score < 350:  # Rock bottom score
        probability = max(probability, 0.93)
    elif application.cibil_score < 450:  # Very bad score
        probability = max(probability, 0.88)
    
    # Rule 3: Cheque Bounces - Criminal offense under Section 138 of Negotiable Instruments Act
    if application.cheque_bounces >= 5:  # 5+ bounces = serious pattern
        probability = max(probability, 0.90)
    elif application.cheque_bounces >= 3:  # 3+ bounces = concerning pattern
        probability = max(probability, 0.80)
    
    # Rule 4: FOIR (Fixed Obligation to Income Ratio) - Mathematical affordability
    if foir > 100:  # Cannot afford (EMI exceeds income)
        probability = max(probability, 0.95)
    elif foir > 80:  # Critically high burden
        probability = max(probability, 0.85)
    
    if probability > original_prob:
        logger.info(f"HARD RULE applied: {original_prob*100:.1f}% → {probability*100:.1f}%")
    
    return probability


# Pydantic models for request/response
class LoanApplication(BaseModel):
    """India-realistic loan application with FO IR, DPD, Banking Behavior"""
    
    # Loan details
    loan_amnt: float = Field(..., gt=0, le=50000000, description="Loan amount (₹) - must be positive, max ₹5Cr", example=500000)
    int_rate: float = Field(..., gt=0, le=50, description="Interest rate (%) - must be 0-50%", example=12.5)
    term: int = Field(..., description="Loan term (months): 12/24/36/48/60", example=36)
    purpose: str = Field(..., description="Loan purpose", example="personal")
    
    # Personal info
    age: int = Field(..., ge=18, le=100, description="Age (18-100)", example=32)
    annual_inc: float = Field(..., gt=0, le=100000000, description="Annual income (₹) - must be positive, max ₹10Cr", example=600000)
    emp_length: int = Field(..., ge=0, le=50, description="Employment length (years) - 0-50", example=5)
    employment_type: str = Field(..., description="Employment: Govt/MNC/Private/Self-employed", example="MNC")
    home_ownership: str = Field(..., description="Home: RENT/OWN/FAMILY", example="RENT")
    city_tier: str = Field(..., description="City: Metro/Tier1/Tier2", example="Metro")
    
    # INDIA-CRITICAL: CIBIL & FOIR
    cibil_score: int = Field(..., ge=300, le=900, description="CIBIL Score (300-900), 750+ is excellent", example=720)
    existing_emi: float = Field(..., ge=0, le=1000000, description="Total existing EMIs (₹/month) - 0 to ₹10L", example=15000)
    
    # Credit bureau data
    dpd_last_12m: int = Field(..., ge=0, le=360, description="Days Past Due in last 12 months (0 = good, max 360)", example=0)
    cheque_bounces: int = Field(..., ge=0, le=50, description="Cheque bounces in last year (0-50)", example=0)
    inq_last_6mths: int = Field(..., ge=0, le=20, description="Credit inquiries in last 6 months (0-20)", example=1)
    num_unsecured_loans: int = Field(..., ge=0, le=20, description="Number of unsecured loans (0-20)", example=2)
    
    # Credit accounts
    open_acc: int = Field(..., ge=0, le=50, description="Open credit accounts (0-50)", example=5)
    total_acc: int = Field(..., ge=0, le=100, description="Total credit accounts (0-100)", example=10)
    revol_bal: float = Field(..., ge=0, le=10000000, description="Credit card balance (₹) - 0 to ₹1Cr", example=50000)
    revol_util: float = Field(..., ge=0, le=100, description="Credit utilization (%) - 0-100%", example=35.0)
    
    @model_validator(mode='after')
    def validate_term_field(self):
        """Validate that term is one of the standard loan durations"""
        if self.term not in [12, 24, 36, 48, 60]:
            raise ValueError(f'Term must be one of: 12, 24, 36, 48, or 60 months. Got: {self.term}')
        return self


class PredictionResponse(BaseModel):
    """Prediction response"""
    default_probability: float
    risk_category: str
    decision: str
    interest_rate: float
    interest_rate_pct: float
    expected_profit: Optional[float]
    confidence_score: float


class ExplanationResponse(BaseModel):
    """Explanation response"""
    prediction: Dict[str, Any]
    top_features: List[Dict[str, Any]]
    counterfactual: Dict[str, Any]


# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global model, preprocessor, explainer, risk_engine, feature_names
    
    logger.info("Loading models...")
    
    try:
        # Get project root (parent of api folder)
        project_root = Path(__file__).parent.parent
        
        # Load model
        model_path = project_root / "models" / "xgboost_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"XGBoost model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}. Predictions will fail until model is trained.")
        
        # Load preprocessor
        preprocessor_path = project_root / "models" / "preprocessor.pkl"
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
            feature_names = preprocessor['feature_names']
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
        # Initialize risk engine
        risk_engine = CreditRiskEngine()
        logger.info("Risk engine initialized")
        
        # Initialize explainer (will be set up when first prediction is made)
        logger.info("API ready")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Risk Assessment API",
        "version": "1.0.0",
        "status": "online",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "risk_engine_loaded": risk_engine is not None
    }


@app.post("/predict", response_model= PredictionResponse)
async def predict(application: LoanApplication):
    """
    Predict default probability for India-realistic loan application
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame([application.dict()])
        
        # Calculate monthly income
        data['monthly_income'] = data['annual_inc'] / 12
        
        # Calculate NEW EMI for this loan
        # EMI = P × r × (1+r)^n / ((1+r)^n - 1)
        monthly_rate = (data['int_rate'] / 100) / 12
        data['installment'] = (
            data['loan_amnt'] * monthly_rate * (1 + monthly_rate) ** data['term']
        ) / ((1 + monthly_rate) ** data['term'] - 1)
        
        # Calculate FOIR - MOST CRITICAL INDIAN METRIC
        # FOIR = (Existing EMIs + New EMI) / Monthly Income * 100
        data['foir'] = (data['existing_emi'] + data['installment']) / data['monthly_income'] * 100
        
        # Add derived features
        data['loan_income_ratio'] = data['loan_amnt'] / data['annual_inc']
        data['emi_income_ratio'] = data['installment'] / data['monthly_income'] * 100
        data['cibil_score_high'] = data['cibil_score'] + 50
        
        # Preprocess
        from preprocessor import CreditRiskPreprocessor
        prep = CreditRiskPreprocessor()
        prep.scaler = preprocessor['scaler']
        prep.label_encoders = preprocessor['label_encoders']
        prep.feature_names = preprocessor['feature_names']
        prep.numeric_features = preprocessor['numeric_features']
        prep.categorical_features = preprocessor['categorical_features']
        
        X = prep.transform(data)
        
        # Predict
        probability = float(model.predict_proba(X)[0, 1])
        
        # Apply HARD BUSINESS RULES (overrides ML when necessary)
        calculated_foir = float(data['foir'].iloc[0])
        probability = apply_hard_business_rules(probability, application, calculated_foir)
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(probability - 0.5) * 2
        
        # Get risk assessment
        assessment = risk_engine.assess_applicant(
            default_probability=probability,
            loan_amount=application.loan_amnt,
            confidence_score=confidence
        )
        
        return PredictionResponse(**assessment)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain(application: LoanApplication):
    """
    Provide SHAP explanation for a prediction
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame([application.dict()])
        
        # Calculate monthly income
        data['monthly_income'] = data['annual_inc'] / 12
        
        # Calculate NEW EMI for this loan
        monthly_rate = (data['int_rate'] / 100) / 12
        data['installment'] = (
            data['loan_amnt'] * monthly_rate * (1 + monthly_rate) ** data['term']
        ) / ((1 + monthly_rate) ** data['term'] - 1)
        
        # Calculate FOIR - MOST CRITICAL INDIAN METRIC
        data['foir'] = (data['existing_emi'] + data['installment']) / data['monthly_income'] * 100
        
        # Add derived features
        data['loan_income_ratio'] = data['loan_amnt'] / data['annual_inc']
        data['emi_income_ratio'] = data['installment'] / data['monthly_income'] * 100
        data['cibil_score_high'] = data['cibil_score'] + 50
        
        # Preprocess
        from preprocessor import CreditRiskPreprocessor
        prep = CreditRiskPreprocessor()
        prep.scaler = preprocessor['scaler']
        prep.label_encoders = preprocessor['label_encoders']
        prep.feature_names = preprocessor['feature_names']
        prep.numeric_features = preprocessor['numeric_features']
        prep.categorical_features = preprocessor['categorical_features']
        
        X = prep.transform(data)
        
        # Initialize explainer if needed
        global explainer
        if explainer is None:
            explainer = CreditRiskExplainer(model, feature_names)
            # Load background data for SHAP
            project_root = Path(__file__).parent.parent
            background_path = project_root / "models" / "shap_background.pkl"
            if background_path.exists():
                X_background = joblib.load(background_path)
            else:
                X_background = X  # Use current instance as fallback
            explainer.initialize_shap(X_background, max_samples=100)
        
        # Generate explanation
        explanation = explainer.generate_full_explanation(X, 0, X)
        
        # Make serializable
        explanation = ensure_serializable(explanation)
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(applications: List[LoanApplication]):
    """
    Predict for multiple applications
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame([app.dict() for app in applications])
        
        # Preprocess
        from preprocessor import CreditRiskPreprocessor
        prep = CreditRiskPreprocessor()
        prep.scaler = preprocessor['scaler']
        prep.label_encoders = preprocessor['label_encoders']
        prep.feature_names = preprocessor['feature_names']
        prep.numeric_features = preprocessor['numeric_features']
        prep.categorical_features = preprocessor['categorical_features']
        
        X = prep.transform(data)
        
        # Predict
        probabilities = model.predict_proba(X)[:, 1]
        loan_amounts = data['loan_amnt'].values
        
        # Assess all
        assessments = risk_engine.batch_assess(probabilities, loan_amounts)
        
        # Calculate portfolio metrics
        portfolio_metrics = risk_engine.calculate_portfolio_metrics(assessments)
        
        return {
            "predictions": ensure_serializable(assessments.to_dict('records')),
            "portfolio_metrics": ensure_serializable(portfolio_metrics)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if feature_names:
                feature_importance = [
                    {"feature": name, "importance": float(imp)}
                    for name, imp in zip(feature_names, importance)
                ]
                feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)
            else:
                feature_importance = []
        else:
            feature_importance = []
        
        # Get model parameters
        params = model.get_params() if hasattr(model, 'get_params') else {}
        
        return {
            "model_type": type(model).__name__,
            "n_features": len(feature_names) if feature_names else 0,
            "feature_names": feature_names[:20] if feature_names else [],  # First 20
            "top_features": feature_importance[:10],  # Top 10
            "parameters": {k: str(v) for k, v in params.items() if k in ['max_depth', 'n_estimators', 'learning_rate']}
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
