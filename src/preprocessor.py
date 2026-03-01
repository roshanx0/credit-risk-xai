"""
Data Preprocessor for Credit Risk Assessment
Handles cleaning, feature engineering, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import Tuple, List, Dict
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskPreprocessor:
    """Preprocessing pipeline for credit risk data"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'default') -> Tuple[pd.DataFrame, pd.Series]:
        """Fit preprocessor and transform data"""
        
        df = df.copy()
        
        # Separate target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        logger.info(f"Starting preprocessing. Shape: {X.shape}")
        
        # 1. Identify feature types
        self._identify_feature_types(X)
        
        # 2. Handle missing values
        X = self._handle_missing_values(X)
        
        # 3. Feature engineering
        X = self._engineer_features(X)
        
        # 4. Outlier handling
        X = self._handle_outliers(X)
        
        # 5. Encode categorical variables
        X = self._encode_categorical(X, fit=True)
        
        # 6. Scale numeric features
        X = self._scale_features(X, fit=True)
        
        # 7. Feature selection
        X = self._select_features(X, y)
        
        self.feature_names = X.columns.tolist()
        logger.info(f"Preprocessing complete. Final shape: {X.shape}")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        
        X = df.copy()
        
        # Drop target if present
        if 'default' in X.columns:
            X = X.drop(columns=['default'])
        
        # Apply transformations
        X = self._handle_missing_values(X)
        X = self._engineer_features(X)
        X = self._handle_outliers(X)
        X = self._encode_categorical(X, fit=False)
        X = self._scale_features(X, fit=False)
        
        # Ensure same features as training
        missing_cols = set(self.feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        
        X = X[self.feature_names]
        
        return X
    
    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify numeric and categorical features"""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric features: {len(self.numeric_features)}")
        logger.info(f"Categorical features: {len(self.categorical_features)}")
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with median/mode imputation"""
        
        # Numeric: median
        if self.numeric_features:
            X[self.numeric_features] = X[self.numeric_features].fillna(
                X[self.numeric_features].median()
            )
        
        # Categorical: mode
        if self.categorical_features:
            X[self.categorical_features] = X[self.categorical_features].fillna(
                X[self.categorical_features].mode().iloc[0]
            )
        
        logger.info(f"Missing values handled. Remaining nulls: {X.isnull().sum().sum()}")
        return X
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create derived features"""
        
        # Debt-to-income ratio (if not already present)
        if 'dti' not in X.columns and 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['dti'] = X['loan_amnt'] / X['annual_inc'] * 100
        
        # Credit utilization (if revol_bal and revol_util available)
        if 'revol_util' in X.columns and 'revol_bal' in X.columns:
            X['credit_utilization_ratio'] = X['revol_util'] / 100
        
        # Loan to income ratio
        if 'loan_amnt' in X.columns and 'annual_inc' in X.columns:
            X['loan_income_ratio'] = X['loan_amnt'] / (X['annual_inc'] + 1)  # +1 to avoid division by zero
        
        # FICO average
        if 'fico_range_low' in X.columns and 'fico_range_high' in X.columns:
            X['fico_avg'] = (X['fico_range_low'] + X['fico_range_high']) / 2
        
        # Risk indicators
        if 'delinq_2yrs' in X.columns:
            X['has_delinquency'] = (X['delinq_2yrs'] > 0).astype(int)
        
        if 'pub_rec' in X.columns:
            X['has_public_record'] = (X['pub_rec'] > 0).astype(int)
        
        logger.info("Feature engineering complete")
        return X
    
    def _handle_outliers(self, X: pd.DataFrame, std_threshold: float = 3) -> pd.DataFrame:
        """Clip outliers beyond n standard deviations"""
        
        for col in self.numeric_features:
            if col in X.columns:
                mean = X[col].mean()
                std = X[col].std()
                
                lower = mean - std_threshold * std
                upper = mean + std_threshold * std
                
                X[col] = X[col].clip(lower, upper)
        
        logger.info(f"Outliers clipped at {std_threshold} std")
        return X
    
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categorical variables"""
        
        for col in self.categorical_features:
            if col not in X.columns:
                continue
                
            if fit:
                # Create label encoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use fitted encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unknown categories
                    X[col] = X[col].astype(str).map(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    X[col] = le.transform(X[col])
        
        logger.info(f"Categorical encoding complete")
        return X
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Scale numeric features"""
        
        numeric_cols = [col for col in self.numeric_features if col in X.columns]
        
        if fit:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        else:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        logger.info("Feature scaling complete")
        return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select relevant features (remove low variance, high correlation)"""
        
        # Remove constant features
        variance = X.var()
        constant_features = variance[variance < 0.01].index.tolist()
        if constant_features:
            X = X.drop(columns=constant_features)
            logger.info(f"Removed {len(constant_features)} constant features")
        
        # Remove highly correlated features
        if len(X.columns) > 1:
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
            if to_drop:
                X = X.drop(columns=to_drop)
                logger.info(f"Removed {len(to_drop)} highly correlated features")
        
        return X
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series, 
                    sampling_strategy: float = 0.5) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for handling class imbalance"""
        
        logger.info(f"Applying SMOTE with sampling_strategy={sampling_strategy}")
        logger.info(f"Before SMOTE: {y.value_counts().to_dict()}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def save(self, path: str):
        """Save preprocessor to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor from disk"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.numeric_features = data['numeric_features']
        self.categorical_features = data['categorical_features']
        logger.info(f"Preprocessor loaded from {path}")


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, 
               val_size: float = 0.1,
               random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """Split data into train, validation, and test sets"""
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }


if __name__ == "__main__":
    # Test preprocessor
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_data()
    
    preprocessor = CreditRiskPreprocessor()
    X, y = preprocessor.fit_transform(df)
    
    print(f"\nPreprocessed shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    
    # Split data
    data_splits = split_data(X, y)
    print(f"\nTrain size: {len(data_splits['X_train'])}")
    print(f"Val size: {len(data_splits['X_val'])}")
    print(f"Test size: {len(data_splits['X_test'])}")
