"""
Explainability Module for Credit Risk Models
Implements SHAP (TreeExplainer) and LIME explanations
"""

import numpy as np
import pandas as pd
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskExplainer:
    """Generate explanations for credit risk predictions"""
    
    def __init__(self, model, feature_names: List[str], 
                 output_dir: str = "outputs/visualizations"):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.shap_values = None
        
    def initialize_shap(self, X_background: pd.DataFrame, max_samples: int = 1000):
        """Initialize SHAP TreeExplainer"""
        
        logger.info("Initializing SHAP TreeExplainer...")
        
        # Use sample for background if dataset is large
        if len(X_background) > max_samples:
            X_background = X_background.sample(n=max_samples, random_state=42)
        
        # Initialize TreeExplainer (fast for tree-based models)
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        logger.info(f"SHAP initialized with {len(X_background)} background samples")
    
    def initialize_lime(self, X_train: pd.DataFrame):
        """Initialize LIME explainer"""
        
        logger.info("Initializing LIME explainer...")
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=self.feature_names,
            mode='classification',
            random_state=42
        )
        
        logger.info("LIME initialized")
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for dataset"""
        
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap() first.")
        
        logger.info(f"Computing SHAP values for {len(X)} samples...")
        
        shap_values = self.shap_explainer.shap_values(X)
        
        # For binary classification, XGBoost returns values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        self.shap_values = shap_values
        
        logger.info("SHAP values computed")
        return shap_values
    
    def plot_shap_summary(self, X: pd.DataFrame, shap_values: np.ndarray = None,
                         max_display: int = 20):
        """Create SHAP summary plot (global importance)"""
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available. Call compute_shap_values() first.")
        
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        save_path = self.output_dir / 'shap_summary_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {save_path}")
        plt.close()
    
    def plot_shap_bar(self, X: pd.DataFrame, shap_values: np.ndarray = None,
                     max_display: int = 20):
        """Create SHAP bar plot (mean absolute importance)"""
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info("Creating SHAP bar plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        
        save_path = self.output_dir / 'shap_bar_plot.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP bar plot saved to {save_path}")
        plt.close()
    
    def plot_shap_force(self, instance_idx: int, X: pd.DataFrame, 
                       shap_values: np.ndarray = None):
        """Create SHAP force plot for a single prediction"""
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info(f"Creating SHAP force plot for instance {instance_idx}...")
        
        # Get expected value (base value)
        expected_value = self.shap_explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Create force plot
        shap.force_plot(
            expected_value,
            shap_values[instance_idx],
            X.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        save_path = self.output_dir / f'shap_force_plot_{instance_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP force plot saved to {save_path}")
        plt.close()
    
    def plot_shap_dependence(self, feature: str, X: pd.DataFrame,
                            shap_values: np.ndarray = None,
                            interaction_feature: str = None):
        """Create SHAP dependence plot showing feature effect"""
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        logger.info(f"Creating SHAP dependence plot for {feature}...")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature, shap_values, X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        
        save_path = self.output_dir / f'shap_dependence_{feature}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP dependence plot saved to {save_path}")
        plt.close()
    
    def explain_instance_lime(self, instance: pd.DataFrame, 
                             num_features: int = 10) -> Dict[str, Any]:
        """Generate LIME explanation for a single instance"""
        
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime() first.")
        
        logger.info("Generating LIME explanation...")
        
        # Get explanation
        exp = self.lime_explainer.explain_instance(
            instance.values.flatten(),
            self.model.predict_proba,
            num_features=num_features
        )
        
        # Extract explanation data
        explanation = {
            'prediction': self.model.predict_proba(instance)[0][1],
            'features': [feat[0] for feat in exp.as_list()],
            'contributions': [feat[1] for feat in exp.as_list()],
            'intercept': exp.intercept[1]
        }
        
        logger.info("LIME explanation generated")
        return explanation
    
    def get_top_features_for_instance(self, instance_idx: int, 
                                     X: pd.DataFrame,
                                     shap_values: np.ndarray = None,
                                     top_n: int = 10) -> pd.DataFrame:
        """Get top contributing features for a specific prediction"""
        
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("No SHAP values available")
        
        # Get SHAP values for this instance
        instance_shap = shap_values[instance_idx]
        instance_features = X.iloc[instance_idx]
        
        # Create dataframe
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': instance_features.values,
            'shap_value': instance_shap,
            'abs_shap': np.abs(instance_shap)
        }).sort_values('abs_shap', ascending=False).head(top_n)
        
        return contributions
    
    def generate_counterfactual(self, instance: pd.DataFrame, X: pd.DataFrame,
                               target_change: float = 0.1,
                               max_features: int = 3) -> Dict[str, Any]:
        """
        Generate counterfactual explanation
        "What would need to change for a different decision?"
        """
        
        logger.info("Generating counterfactual explanation...")
        
        # Get current prediction
        current_prob = self.model.predict_proba(instance)[0][1]
        
        # Get feature importance for this instance
        shap_values = self.shap_explainer.shap_values(instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Find top features to modify
        feature_importance = np.abs(shap_values[0])
        top_features_idx = np.argsort(feature_importance)[::-1][:max_features]
        
        suggestions = []
        for idx in top_features_idx:
            feature_name = self.feature_names[idx]
            current_value = instance.iloc[0, idx]
            shap_contribution = shap_values[0, idx]
            
            # Suggest change direction
            if shap_contribution > 0:  # Increases default risk
                suggested_direction = "decrease"
                # Find similar instances with lower values
                lower_values = X[X.iloc[:, idx] < current_value].iloc[:, idx]
                if len(lower_values) > 0:
                    suggested_value = lower_values.median()
                else:
                    suggested_value = current_value * 0.8
            else:  # Decreases default risk
                suggested_direction = "increase"
                # Find similar instances with higher values
                higher_values = X[X.iloc[:, idx] > current_value].iloc[:, idx]
                if len(higher_values) > 0:
                    suggested_value = higher_values.median()
                else:
                    suggested_value = current_value * 1.2
            
            suggestions.append({
                'feature': feature_name,
                'current_value': float(current_value),
                'suggested_value': float(suggested_value),
                'direction': suggested_direction,
                'impact': float(abs(shap_contribution))
            })
        
        counterfactual = {
            'current_probability': float(current_prob),
            'target_probability': float(current_prob - target_change if current_prob > 0.5 else current_prob + target_change),
            'suggestions': suggestions
        }
        
        logger.info("Counterfactual generated")
        return counterfactual
    
    def generate_full_explanation(self, instance: pd.DataFrame, 
                                 instance_idx: int,
                                 X_background: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive explanation combining SHAP and other methods"""
        
        logger.info("Generating full explanation...")
        
        # Get prediction
        prediction_proba = self.model.predict_proba(instance)[0][1]
        prediction_class = int(prediction_proba >= 0.5)
        
        # Compute SHAP for this instance
        shap_values = self.shap_explainer.shap_values(instance)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get top features
        top_features = self.get_top_features_for_instance(
            0, instance, shap_values, top_n=10
        )
        
        # Generate counterfactual
        counterfactual = self.generate_counterfactual(instance, X_background)
        
        explanation = {
            'prediction': {
                'probability': float(prediction_proba),
                'class': prediction_class,
                'risk_category': 'HIGH' if prediction_proba >= 0.7 else ('MODERATE' if prediction_proba >= 0.3 else 'LOW')
            },
            'top_features': top_features.to_dict('records'),
            'counterfactual': counterfactual,
            'shap_values': shap_values.tolist()
        }
        
        logger.info("Full explanation generated")
        return explanation
    
    def save(self, path: str):
        """Save explainer configuration"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'feature_names': self.feature_names,
            'shap_values': self.shap_values
        }, path)
        logger.info(f"Explainer saved to {path}")


if __name__ == "__main__":
    # Test explainer
    from data_loader import DataLoader
    from preprocessor import CreditRiskPreprocessor, split_data
    from model_trainer import CreditRiskModelTrainer
    
    # Load data and train model
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = CreditRiskPreprocessor()
    X, y = preprocessor.fit_transform(df)
    data_splits = split_data(X, y)
    
    # Train quick model
    trainer = CreditRiskModelTrainer()
    model = trainer.train_xgboost(
        data_splits['X_train'], data_splits['y_train'],
        data_splits['X_val'], data_splits['y_val'],
        param_grid={'max_depth': [4], 'n_estimators': [100], 'learning_rate': [0.1]},
        cv=2
    )
    
    # Initialize explainer
    explainer = CreditRiskExplainer(model, X.columns.tolist())
    explainer.initialize_shap(data_splits['X_train'])
    
    # Generate explanations
    shap_values = explainer.compute_shap_values(data_splits['X_test'])
    explainer.plot_shap_summary(data_splits['X_test'])
    explainer.plot_shap_bar(data_splits['X_test'])
    
    print("Explainer test complete!")
