"""
Data Loader for Credit Risk Dataset
Downloads and loads Lending Club dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handle data downloading and loading"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_lending_club(self, sample_size: Optional[int] = 50000) -> pd.DataFrame:
        """
        Generate Lending Club-style dataset for demonstration
        """
        logger.info("Generating Lending Club-style dataset...")
        
        # Generate synthetic data directly
        df = self._create_synthetic_data(sample_size or 10000)
        
        return df
    
    def _create_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Create India-realistic credit risk dataset
        Based on actual Indian lending practices: CIBIL, FOIR, Employment, Banking Behavior
        """
        np.random.seed(42)
        
        logger.info(f"Generating {n_samples} INDIA-REALISTIC loan applications...")
        
        # 1. CIBIL Score (Most important in India - explains ~40% of decision)
        # Realistic distribution: bell curve centered around 700
        cibil_score = np.random.normal(700, 80, n_samples).astype(int)
        cibil_score = np.clip(cibil_score, 300, 900)
        
        # 2. Demographics
        age = np.random.randint(23, 65, n_samples)
        employment_type = np.random.choice(
            ['Govt', 'MNC', 'Private', 'Self-employed'],
            n_samples,
            p=[0.15, 0.30, 0.35, 0.20]  # MNC/Private most common
        )
        city_tier = np.random.choice(
            ['Metro', 'Tier1', 'Tier2'],
            n_samples,
            p=[0.45, 0.35, 0.20]
        )
        
        # 3. Income (employment type affects income)
        base_income = np.random.lognormal(14.5, 0.6, n_samples)  # Base ₹3L-₹30L
        income_multiplier = np.where(employment_type == 'Govt', 1.1,
                             np.where(employment_type == 'MNC', 1.3,
                             np.where(employment_type == 'Private', 0.9, 1.0)))
        annual_inc = base_income * income_multiplier
        annual_inc = np.clip(annual_inc, 200000, 50000000)  # ₹2L - ₹5Cr
        monthly_income = annual_inc / 12
        
        # 4. Employment Length (years in current job)
        emp_length = np.random.randint(0, 15, n_samples)
        # Govt jobs = more stable
        emp_length = np.where(employment_type == 'Govt', emp_length + 3, emp_length)
        emp_length = np.clip(emp_length, 0, 30)
        
        # 5. Loan Details
        loan_amnt = np.random.randint(50000, 5000000, n_samples)  # ₹50k - ₹50L
        term = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.35, 0.2, 0.15])
        
        # Interest rate based on CIBIL (risk-based pricing in India)
        int_rate = np.where(cibil_score >= 750, np.random.uniform(9, 12, n_samples),
                   np.where(cibil_score >= 650, np.random.uniform(12, 16, n_samples),
                   np.random.uniform(16, 22, n_samples)))
        
        # 6. EXISTING EMI (other loan EMIs) - Critical for FOIR
        # More loans for higher income, worse CIBIL = more debt
        existing_emi_base = monthly_income * np.random.uniform(0.1, 0.4, n_samples)
        cibil_risk = (900 - cibil_score) / 600  # 0-1 (lower CIBIL = more debt)
        existing_emi = existing_emi_base * (1 + cibil_risk * 0.5)
        existing_emi = np.clip(existing_emi, 0, monthly_income * 0.6)
        
        # 7. Calculate NEW EMI for this loan
        monthly_rate = (int_rate / 100) / 12
        new_emi = (loan_amnt * monthly_rate * (1 + monthly_rate) ** term) / ((1 + monthly_rate) ** term - 1)
        
        # 8. FOIR - MOST CRITICAL METRIC IN INDIA
        # FOIR = (Existing EMIs + New EMI) / Monthly Income
        foir = (existing_emi + new_emi) / monthly_income * 100
        
        # 9. Credit Bureau Data (Indian-specific)
        # DPD = Days Past Due (critical metric for repayment history)
        # Increased probability of high DPD for better model learning
        dpd_last_12m = np.random.choice([0, 0, 0, 1, 2, 3, 7, 15, 30, 60, 90], n_samples,
                                        p=[0.55, 0.12, 0.10, 0.08, 0.05, 0.03, 0.03, 0.02, 0.01, 0.0075, 0.0025])
        # DPD worse for low CIBIL (more likely to have payment issues)
        dpd_last_12m = np.where(cibil_score < 650, dpd_last_12m + np.random.poisson(3, n_samples), dpd_last_12m)
        dpd_last_12m = np.clip(dpd_last_12m, 0, 180)
        
        # 10. Banking Behavior (bounces, cheques)
        # Cheque bounces = criminal offense under Section 138 if > 1
        cheque_bounces = np.random.poisson(0.3, n_samples)
        cheque_bounces = np.where(cibil_score < 650, cheque_bounces + 2, cheque_bounces)
        cheque_bounces = np.where(dpd_last_12m > 30, cheque_bounces + 1, cheque_bounces)  # DPD linked to bounces
        cheque_bounces = np.clip(cheque_bounces, 0, 10)
        
        # 11. Credit History
        total_acc = np.random.randint(2, 20, n_samples)
        open_acc = np.random.randint(1, total_acc + 1)
        num_unsecured_loans = np.random.randint(0, 5, n_samples)
        
        # Credit inquiries (too many = credit hungry)
        inq_last_6mths = np.random.poisson(0.8, n_samples)
        inq_last_6mths = np.where(cibil_score < 650, inq_last_6mths + 2, inq_last_6mths)
        inq_last_6mths = np.clip(inq_last_6mths, 0, 10)
        
        # 12. Other factors
        home_ownership = np.random.choice(['RENT', 'OWN', 'FAMILY'], n_samples, p=[0.35, 0.35, 0.3])
        purpose = np.random.choice([
            'personal', 'business', 'education', 'home_renovation',
            'medical', 'wedding', 'vehicle', 'debt_consolidation'
        ], n_samples, p=[0.25, 0.15, 0.1, 0.15, 0.05, 0.1, 0.1, 0.1])
        
        # Credit card utilization
        revol_util = np.random.uniform(10, 95, n_samples)
        revol_util = np.where(cibil_score >= 750, revol_util * 0.6, revol_util)  # Good CIBIL = lower util
        revol_bal = (annual_inc * 0.3) * (revol_util / 100)
        
        # ============================================================
        # Generate DEFAULT TARGET using INDIA-REALISTIC RULES
        # ============================================================
        
        # Normalize key features
        cibil_norm = (900 - cibil_score) / 600  # 0-1 (lower = riskier)
        
        # FOIR normalization - DON'T CLIP! FOIR > 100% should have exponential impact
        foir_norm = foir / 65  # Normalize to 65% (ideal threshold)
        # Above 100% FOIR = mathematically impossible to afford
        foir_norm = np.where(foir > 100, foir_norm * 2, foir_norm)  # Double penalty for FOIR > 100%
        
        # DPD normalization - DON'T CLIP! DPD > 30 should have exponential impact
        # DPD is THE most critical metric for payment history
        dpd_norm = dpd_last_12m / 30  # Normalize to 30 days (threshold)
        dpd_norm = np.where(dpd_last_12m > 60, dpd_norm * 1.5, dpd_norm)  # 60+ days = 1.5x penalty
        dpd_norm = np.where(dpd_last_12m > 90, dpd_norm * 2.0, dpd_norm)  # 90+ days = 3x total penalty
        
        # Income risk - Lower income = higher risk (even with same FOIR)
        # ₹2L income = 1.0 risk, ₹20L income = 0.0 risk
        income_risk = np.clip((5000000 - annual_inc) / 4800000, 0, 1)
        
        # Term risk - Shorter term = higher monthly burden (riskier short-term)
        # But also shows capacity, so moderate impact
        term_risk = np.where(term <= 24, 0.08,  # 12-24 months = slight risk
                    np.where(term == 36, 0.0,   # 36 months = baseline
                    -0.05))  # 48-60 months = slightly safer (lower EMI)
        
        # INDIAN LENDING RULE: CIBIL + FOIR + DPD explain ~75% of decision
        default_prob = (
            -0.50 +  # Base
            cibil_norm * 0.55 +  # CIBIL is KING (40% weight)
            foir_norm * 0.50 +  # FOIR is second (35% weight)
            dpd_norm * 0.45 +  # DPD is third most critical (20% weight) - INCREASED from 0.35
            income_risk * 0.12 +  # Direct income effect (10% weight)
            term_risk +  # Term impact (3% weight)
            (cheque_bounces > 0).astype(int) * 0.22 +  # Banking issues
            (cheque_bounces >= 3).astype(int) * 0.30 +  # Multiple bounces = serious (additional)
            (num_unsecured_loans >= 3).astype(int) * 0.20 +  # Too many unsecured
            (inq_last_6mths >= 3).astype(int) * 0.15 +  # Credit hungry
            (revol_util > 70).astype(int) * 0.20 +  # High credit card usage
            (employment_type == 'Self-employed').astype(int) * 0.10  # Self-employed risk
        )
        
        # Interaction effects (realistic combinations)
        default_prob += ((cibil_norm > 0.5) & (foir > 60)).astype(int) * 0.40  # Bad CIBIL + High FOIR = RED FLAG
        default_prob += ((dpd_last_12m > 0) & (cibil_norm > 0.4)).astype(int) * 0.30  # DPD + Low CIBIL
        default_prob += ((cheque_bounces > 1) & (cibil_norm > 0.3)).astype(int) * 0.25  # Bounces + Bad credit
        
        # Employment type modifiers
        default_prob -= (employment_type == 'Govt').astype(int) * 0.15  # Govt job = safer
        default_prob -= (employment_type == 'MNC').astype(int) * 0.08  # MNC = stable
        
        # HARD RULES - Banking reality (these override calculations above)
        # FOIR > 65% = High risk, banks rarely approve
        default_prob = np.where(foir > 65, np.maximum(default_prob, 0.73), default_prob)
        
        # FOIR > 100% = Mathematically impossible, guaranteed default
        default_prob = np.where(foir > 100, np.maximum(default_prob, 0.95), default_prob)
        
        # CIBIL thresholds
        default_prob = np.where(cibil_score < 600, np.maximum(default_prob, 0.78), default_prob)  # Sub-prime
        default_prob = np.where(cibil_score < 450, np.maximum(default_prob, 0.88), default_prob)  # Very bad
        default_prob = np.where(cibil_score < 350, np.maximum(default_prob, 0.93), default_prob)  # Rock bottom
        
        # DPD thresholds - CRITICAL! Payment history is reality, not prediction
        default_prob = np.where(dpd_last_12m > 30, np.maximum(default_prob, 0.72), default_prob)  # 30+ days
        default_prob = np.where(dpd_last_12m > 60, np.maximum(default_prob, 0.85), default_prob)  # 60+ days (NPA soon)
        default_prob = np.where(dpd_last_12m > 90, np.maximum(default_prob, 0.93), default_prob)  # 90+ days (NPA)
        
        # Cheque bounces - Criminal offense
        default_prob = np.where(cheque_bounces >= 3, np.maximum(default_prob, 0.80), default_prob)  # 3+ bounces
        default_prob = np.where(cheque_bounces >= 5, np.maximum(default_prob, 0.90), default_prob)  # 5+ bounces
        
        # Clip to realistic range (keep some uncertainty)
        default_prob = np.clip(default_prob, 0.02, 0.98)
        
        # Generate binary target
        default = (np.random.random(n_samples) < default_prob).astype(int)
        
        # ============================================================
        # Create DataFrame with ALL features
        # ============================================================
        df = pd.DataFrame({
            # Loan details
            'loan_amnt': loan_amnt,
            'int_rate': int_rate,
            'term': term,
            'installment': new_emi,  # New EMI for this loan
            'purpose': purpose,
            
            # Personal info
            'age': age,
            'annual_inc': annual_inc,
            'monthly_income': monthly_income,
            'emp_length': emp_length,
            'employment_type': employment_type,
            'home_ownership': home_ownership,
            'city_tier': city_tier,
            
            # INDIA-SPECIFIC: MOST IMPORTANT
            'cibil_score': cibil_score,
            'existing_emi': existing_emi,
            'foir': foir,  # CRITICAL: (Existing EMI + New EMI) / Monthly Income
            
            # Credit bureau data
            'dpd_last_12m': dpd_last_12m,  # Days Past Due (DPD)
            'cheque_bounces': cheque_bounces,
            'inq_last_6mths': inq_last_6mths,
            'num_unsecured_loans': num_unsecured_loans,
            
            # Credit accounts
            'open_acc': open_acc,
            'total_acc': total_acc,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            
            # Target
            'default': default
        })
        
        # Derived features (engineered)
        df['loan_income_ratio'] = df['loan_amnt'] / df['annual_inc']
        df['emi_income_ratio'] = df['installment'] / df['monthly_income'] * 100
        df['cibil_score_high'] = df['cibil_score'] + 50  # Score range
        
        # Save to disk
        output_path = self.data_dir / "lending_club.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ India-realistic dataset generated!")
        logger.info(f"   Default rate: {df['default'].mean():.2%}")
        logger.info(f"   Avg CIBIL: {df['cibil_score'].mean():.0f}")
        logger.info(f"   Avg FOIR: {df['foir'].mean():.1f}%")
        logger.info(f"   Saved to: {output_path}")
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """Load data from disk"""
        data_path = self.data_dir / "lending_club.csv"
        
        if not data_path.exists():
            logger.info("Data not found. Downloading...")
            return self.download_lending_club()
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def get_feature_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary information about features"""
        info = pd.DataFrame({
            'dtype': df.dtypes,
            'missing': df.isnull().sum(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
            'unique': df.nunique(),
            'sample': df.iloc[0]
        })
        return info


if __name__ == "__main__":
    # Test data loader
    loader = DataLoader()
    df = loader.load_data()
    
    print("\n=== Dataset Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nTarget distribution:")
    print(df['default'].value_counts(normalize=True))
    
    print("\n=== Feature Info ===")
    print(loader.get_feature_info(df))
