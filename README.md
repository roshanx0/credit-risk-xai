# India-Realistic Credit Risk Assessment System

A production-ready machine learning system for credit default prediction with explainable AI, designed specifically for the Indian credit market.

## 🎯 Features

- **India-Specific**: CIBIL scores (300-900), FOIR calculation, DPD tracking, cheque bounce detection
- **High Accuracy**: 92.84% ROC-AUC with XGBoost
- **Hybrid System**: Machine Learning + Hard Business Rules (RBI compliant)
- **Explainable AI**: SHAP visualizations for every prediction
- **Risk Engine**: Probability scoring → Risk categorization → Decision rules → Interest rates (8-18%)
- **Input Validation**: Pydantic validators prevent invalid data
- **Interactive Dashboard**: Real-time predictions with 24 input fields
- **RESTful API**: FastAPI backend with automatic documentation

## 📊 Architecture

```
Data → Preprocessing → Model Training → Explainability → Risk Engine → API/Frontend
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Generate Data & Train Model

```bash
python train_quick.py
```

⏱️ Takes 2-3 minutes. Generates 50,000 synthetic loan applications and trains XGBoost model.

### 6. Start API

```bash
python -m uvicorn api.main:app --reload
```

### 7. Open Frontend

Open `frontend/index.html` in your browser or visit http://localhost:8000

## 📁 Project Structure

```
credit-risk-xai-2/
├── data/              # Raw and processed datasets (gitignored)
├── notebooks/         # Jupyter notebook for experimentation
├── src/               # Core Python modules
│   ├── data_loader.py       # Synthetic data generation
│   ├── preprocessor.py      # Feature engineering
│   ├── model_trainer.py     # XGBoost training
│   ├── explainer.py         # SHAP interpretability
│   ├── risk_engine.py       # Risk scoring & decisions
│   └── utils.py             # Helper functions
├── api/               # FastAPI backend
│   └── main.py              # API endpoints + validation
├── frontend/          # Web dashboard (HTML/JS)
├── models/            # Trained models (gitignored)
├── outputs/           # Visualizations and reports
├── config/            # Configuration files
├── tests/             # Unit tests
├── train_quick.py     # Generate data + train model
└── train_better.py    # Slower, more thorough training
```

## 🔬 Model Performance

| Metric    | Score  |
| --------- | ------ |
| ROC-AUC   | 92.84% |
| Precision | 93.02% |
| Recall    | 85.46% |
| F1-Score  | 89.08% |

**Model**: XGBoost (max_depth=5, n_estimators=200, learning_rate=0.05)  
**Training**: 50,000 synthetic India-realistic loan applications  
**Explainability**: SHAP TreeExplainer with 500 background samples

## 🇮🇳 Indian Banking Features

### Credit Bureau

- **CIBIL Score**: 300-900 range (TransUnion CIBIL)
- **FOIR**: Fixed Obligation to Income Ratio = (EMI / Monthly Income) × 100

### Banking Behavior

- **DPD**: Days Past Due (0, 30, 60, 90+ days)
- **Cheque Bounces**: Section 138 NI Act violations
- **Employment Types**: Government, MNC, Private, Self-Employed

### Regulatory Compliance (RBI)

- DPD > 90 days → NPA classification (85% min default)
- CIBIL < 350 → High risk (93% min default)
- Cheque bounces ≥ 5 → Criminal offense (90% min default)
- FOIR > 100% → Unaffordable loan (95% min default)

### Risk-Based Interest Rates

- **LOW Risk** (<30%): 8% per annum
- **MODERATE Risk** (30-70%): 8-13% per annum (scaled)
- **HIGH Risk** (>70%): 18% per annum

## �️ Technologies

- **ML Framework**: XGBoost 2.1.0
- **Explainability**: SHAP 0.45.0
- **API**: FastAPI 0.110.0 + Uvicorn 0.29.0
- **Validation**: Pydantic 2.11.7
- **Data**: Pandas, NumPy, Scikit-learn
- **Frontend**: Pure HTML/CSS/JavaScript

## 📝 License

Educational project - March 2026
