"""
Quick training script for Credit Risk Assessment
Trains models with reduced parameters for faster iteration
"""

import sys
sys.path.append('src')

from data_loader import DataLoader
from preprocessor import CreditRiskPreprocessor, split_data
from model_trainer import CreditRiskModelTrainer
from explainer import CreditRiskExplainer
from risk_engine import CreditRiskEngine
import joblib

print("="*60)
print("Credit Risk Assessment - Quick Training")
print("="*60)

# 1. Load Data
print("\n[1/6] Loading data...")
loader = DataLoader()
df = loader.load_data()
print(f"✓ Loaded {len(df)} samples")

# 2. Preprocess
print("\n[2/6] Preprocessing...")
preprocessor = CreditRiskPreprocessor()
X, y = preprocessor.fit_transform(df, target_col='default')
print(f"✓ Preprocessed to {X.shape[1]} features")

# Split data
data_splits = split_data(X, y, test_size=0.2, val_size=0.1, random_state=42)
print(f"✓ Split: Train={len(data_splits['X_train'])}, Val={len(data_splits['X_val'])}, Test={len(data_splits['X_test'])}")

# 3. Train XGBoost (Primary Model)
print("\n[3/6] Training XGBoost (Primary Model)...")
trainer = CreditRiskModelTrainer()

# Balanced quick training parameters (better than before)
param_grid = {
    'max_depth': [5, 6],           # Deeper for credit patterns
    'n_estimators': [200, 300],    # More trees
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8],
}

xgb_model = trainer.train_xgboost(
    data_splits['X_train'], data_splits['y_train'],
    data_splits['X_val'], data_splits['y_val'],
    param_grid=param_grid,
    cv=3  # Better CV
)
print("✓ XGBoost trained")

# 4. Generate SHAP Explanations
print("\n[4/6] Generating SHAP explanations...")
explainer = CreditRiskExplainer(xgb_model, X.columns.tolist())
explainer.initialize_shap(data_splits['X_train'][:1000], max_samples=500)

# Compute SHAP for test set sample
shap_values = explainer.compute_shap_values(data_splits['X_test'][:1000])
print("✓ SHAP values computed")

# Generate visualizations
explainer.plot_shap_summary(data_splits['X_test'][:1000], max_display=20)
explainer.plot_shap_bar(data_splits['X_test'][:1000], max_display=20)
print("✓ SHAP visualizations saved to outputs/visualizations/")

# 5. Test Risk Engine
print("\n[5/6] Testing risk engine...")
risk_engine = CreditRiskEngine()

# Test predictions
test_probs = xgb_model.predict_proba(data_splits['X_test'][:100])[:, 1]
assessments = risk_engine.batch_assess(test_probs)

print("\nSample Assessments:")
print(assessments.head(5).to_string())

metrics = risk_engine.calculate_portfolio_metrics(assessments)
print(f"\nPortfolio Metrics:")
print(f"  Approval Rate: {metrics['decisions']['approval_rate']:.2%}")
print(f"  Avg Default Prob: {metrics['averages']['default_probability']:.2%}")

# 6. Save Artifacts
print("\n[6/6] Saving artifacts...")

# Save preprocessor
preprocessor.save('models/preprocessor.pkl')

# Save SHAP background
background_sample = data_splits['X_train'][:1000]
joblib.dump(background_sample, 'models/shap_background.pkl')

# Save explainer config
explainer.save('models/explainer.pkl')

print("✓ All artifacts saved to models/")

print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)

print("\n📊 Model Performance:")
print(f"  ROC-AUC: {trainer.results['xgboost']['val']['roc_auc']:.4f}")
print(f"  Precision: {trainer.results['xgboost']['val']['precision']:.4f}")
print(f"  Recall: {trainer.results['xgboost']['val']['recall']:.4f}")
print(f"  F1-Score: {trainer.results['xgboost']['val']['f1']:.4f}")

print("\n🚀 Next Steps:")
print("  1. Start API:")
print("     uvicorn api.main:app --reload")
print("\n  2. Open frontend:")
print("     Open frontend/index.html in your browser")
print("\n  3. Test prediction:")
print("     curl -X POST http://localhost:8000/health")
