"""
Model Trainer for Credit Risk Assessment
Trains XGBoost (primary), LightGBM, and CatBoost models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Train and evaluate credit risk models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame = None, y_val: pd.Series = None,
                      param_grid: dict = None, cv: int = 3) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning"""
        
        logger.info("=" * 50)
        logger.info("Training XGBoost (Primary Model)")
        logger.info("=" * 50)
        
        # Default parameter grid (Paper 1: max_depth=4 was optimal)
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 4, 5],
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
        
        # Base model
        base_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Handle imbalance
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='roc_auc',
            verbose=1, n_jobs=-1
        )
        
        # Fit
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best XGBoost params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        # Evaluate
        metrics = self._evaluate_model(best_model, X_train, y_train, X_val, y_val, "XGBoost")
        
        # Store
        self.models['xgboost'] = best_model
        self.results['xgboost'] = metrics
        
        # Save
        self._save_model(best_model, 'xgboost_model.pkl')
        
        return best_model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None,
                       param_grid: dict = None, cv: int = 3) -> lgb.LGBMClassifier:
        """Train LightGBM model"""
        
        logger.info("=" * 50)
        logger.info("Training LightGBM (Comparison Model)")
        logger.info("=" * 50)
        
        if param_grid is None:
            param_grid = {
                'max_depth': [3, 4, 5],
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 50],
            }
        
        base_model = lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            class_weight='balanced'
        )
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='roc_auc',
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best LightGBM params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        metrics = self._evaluate_model(best_model, X_train, y_train, X_val, y_val, "LightGBM")
        
        self.models['lightgbm'] = best_model
        self.results['lightgbm'] = metrics
        
        self._save_model(best_model, 'lightgbm_model.pkl')
        
        return best_model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame = None, y_val: pd.Series = None,
                       param_grid: dict = None, cv: int = 3) -> cb.CatBoostClassifier:
        """Train CatBoost model"""
        
        logger.info("=" * 50)
        logger.info("Training CatBoost (Comparison Model)")
        logger.info("=" * 50)
        
        if param_grid is None:
            param_grid = {
                'depth': [3, 4, 5],
                'iterations': [100, 200],
                'learning_rate': [0.05, 0.1],
            }
        
        base_model = cb.CatBoostClassifier(
            random_state=42,
            verbose=False,
            auto_class_weights='Balanced'
        )
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='roc_auc',
            verbose=1, n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best CatBoost params: {grid_search.best_params_}")
        logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
        metrics = self._evaluate_model(best_model, X_train, y_train, X_val, y_val, "CatBoost")
        
        self.models['catboost'] = best_model
        self.results['catboost'] = metrics
        
        self._save_model(best_model, 'catboost_model.pkl')
        
        return best_model
    
    def _evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series, 
                       model_name: str) -> Dict[str, Any]:
        """Evaluate model performance"""
        
        metrics = {'train': {}, 'val': {}}
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        
        metrics['train']['roc_auc'] = roc_auc_score(y_train, y_train_proba)
        metrics['train']['precision'] = precision_score(y_train, y_train_pred)
        metrics['train']['recall'] = recall_score(y_train, y_train_pred)
        metrics['train']['f1'] = f1_score(y_train, y_train_pred)
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            
            metrics['val']['roc_auc'] = roc_auc_score(y_val, y_val_proba)
            metrics['val']['precision'] = precision_score(y_val, y_val_pred)
            metrics['val']['recall'] = recall_score(y_val, y_val_pred)
            metrics['val']['f1'] = f1_score(y_val, y_val_pred)
            metrics['val']['confusion_matrix'] = confusion_matrix(y_val, y_val_pred)
            
            logger.info(f"\n{model_name} Validation Results:")
            logger.info(f"ROC-AUC: {metrics['val']['roc_auc']:.4f}")
            logger.info(f"Precision: {metrics['val']['precision']:.4f}")
            logger.info(f"Recall: {metrics['val']['recall']:.4f}")
            logger.info(f"F1-Score: {metrics['val']['f1']:.4f}")
            logger.info(f"\nConfusion Matrix:\n{metrics['val']['confusion_matrix']}")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models"""
        
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 60)
        
        comparison = []
        for model_name, metrics in self.results.items():
            if 'val' in metrics:
                comparison.append({
                    'Model': model_name.upper(),
                    'ROC-AUC': metrics['val']['roc_auc'],
                    'Precision': metrics['val']['precision'],
                    'Recall': metrics['val']['recall'],
                    'F1-Score': metrics['val']['f1']
                })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = self.models_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nComparison saved to {comparison_path}")
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any]:
        """Get the best performing model"""
        
        best_score = 0
        best_model_name = None
        
        for model_name, metrics in self.results.items():
            if 'val' in metrics:
                score = metrics['val'][metric]
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
        
        logger.info(f"Best model: {best_model_name.upper()} ({metric}={best_score:.4f})")
        
        return best_model_name, self.models[best_model_name]
    
    def plot_feature_importance(self, model_name: str = 'xgboost', 
                               feature_names: list = None,
                               top_n: int = 20):
        """Plot feature importance"""
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning(f"Model {model_name} doesn't have feature_importances_")
            return
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'{model_name.upper()} - Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save
        save_path = self.models_dir / f'{model_name}_feature_importance.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    def _save_model(self, model, filename: str):
        """Save model to disk"""
        path = self.models_dir / filename
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, filename: str):
        """Load model from disk"""
        path = self.models_dir / filename
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    # Test model trainer
    from data_loader import DataLoader
    from preprocessor import CreditRiskPreprocessor, split_data
    
    # Load and preprocess data
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = CreditRiskPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    data_splits = split_data(X, y)
    
    # Train models
    trainer = CreditRiskModelTrainer()
    
    # Quick training with reduced grid for testing
    param_grid = {
        'max_depth': [4],
        'n_estimators': [100],
        'learning_rate': [0.1]
    }
    
    trainer.train_xgboost(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val'],
        param_grid=param_grid, cv=2
    )
    
    # Compare and get best
    trainer.compare_models()
    best_model_name, best_model = trainer.get_best_model()
