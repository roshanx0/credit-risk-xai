"""
Risk Engine for Credit Risk Assessment
Converts predictions to business decisions and interest rates
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskEngine:
    """Business logic for credit risk decisions"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize risk engine with thresholds and business rules
        
        Args:
            config: Configuration dict with thresholds and costs
        """
        if config is None:
            config = {
                'thresholds': {
                    'low_risk': 0.3,
                    'high_risk': 0.7
                },
                'interest_rates': {
                    'base_rate': 0.08,
                    'low_risk_adjustment': 0.0,
                    'moderate_risk_adjustment': 0.05,
                    'high_risk_adjustment': 0.10
                },
                'business_costs': {
                    'false_positive_cost': 5000,
                    'false_negative_cost': 20000
                }
            }
        
        self.thresholds = config['thresholds']
        self.interest_rates = config['interest_rates']
        self.business_costs = config['business_costs']
        
        logger.info("Risk Engine initialized")
        logger.info(f"Low risk threshold: {self.thresholds['low_risk']}")
        logger.info(f"High risk threshold: {self.thresholds['high_risk']}")
    
    def categorize_risk(self, default_probability: float) -> str:
        """
        Categorize risk level based on default probability
        
        Returns: 'LOW', 'MODERATE', or 'HIGH'
        """
        if default_probability < self.thresholds['low_risk']:
            return 'LOW'
        elif default_probability < self.thresholds['high_risk']:
            return 'MODERATE'
        else:
            return 'HIGH'
    
    def make_decision(self, default_probability: float, 
                     risk_category: str = None) -> str:
        """
        Make loan approval decision
        
        Returns: 'APPROVED', 'REVIEW', or 'REJECTED'
        """
        if risk_category is None:
            risk_category = self.categorize_risk(default_probability)
        
        if risk_category == 'LOW':
            return 'APPROVED'
        elif risk_category == 'MODERATE':
            return 'REVIEW'  # Manual review required
        else:
            return 'REJECTED'
    
    def calculate_interest_rate(self, default_probability: float,
                               risk_category: str = None) -> float:
        """
        Calculate interest rate based on risk
        
        Returns: Interest rate (e.g., 0.08 for 8%)
        """
        if risk_category is None:
            risk_category = self.categorize_risk(default_probability)
        
        base_rate = self.interest_rates['base_rate']
        
        if risk_category == 'LOW':
            adjustment = self.interest_rates['low_risk_adjustment']
        elif risk_category == 'MODERATE':
            # Scale adjustment based on probability within moderate range
            low_threshold = self.thresholds['low_risk']
            high_threshold = self.thresholds['high_risk']
            risk_ratio = (default_probability - low_threshold) / (high_threshold - low_threshold)
            adjustment = risk_ratio * self.interest_rates['moderate_risk_adjustment']
        else:
            adjustment = self.interest_rates['high_risk_adjustment']
        
        return base_rate + adjustment
    
    def calculate_expected_profit(self, default_probability: float,
                                 loan_amount: float,
                                 interest_rate: float = None,
                                 loan_term_years: int = 3) -> float:
        """
        Calculate expected profit from loan
        
        Args:
            default_probability: Probability of default (0-1)
            loan_amount: Loan amount
            interest_rate: Interest rate (if None, calculated from risk)
            loan_term_years: Loan duration in years
        
        Returns: Expected profit
        """
        if interest_rate is None:
            interest_rate = self.calculate_interest_rate(default_probability)
        
        # Expected revenue (interest payments)
        expected_revenue = loan_amount * interest_rate * loan_term_years
        
        # Expected loss (probability of default * loan amount)
        expected_loss = default_probability * loan_amount
        
        # Expected profit
        expected_profit = expected_revenue - expected_loss
        
        return expected_profit
    
    def assess_applicant(self, default_probability: float,
                        loan_amount: float = None,
                        confidence_score: float = None) -> Dict[str, Any]:
        """
        Complete risk assessment for a loan applicant
        
        Returns: Comprehensive risk assessment dict
        """
        # Risk categorization
        risk_category = self.categorize_risk(default_probability)
        
        # Decision
        decision = self.make_decision(default_probability, risk_category)
        
        # Interest rate
        interest_rate = self.calculate_interest_rate(default_probability, risk_category)
        
        # Expected profit (if loan amount provided)
        expected_profit = None
        if loan_amount is not None:
            expected_profit = self.calculate_expected_profit(
                default_probability, loan_amount, interest_rate
            )
        
        assessment = {
            'default_probability': round(default_probability, 4),
            'risk_category': risk_category,
            'decision': decision,
            'interest_rate': round(interest_rate, 4),
            'interest_rate_pct': round(interest_rate * 100, 2),
            'expected_profit': round(expected_profit, 2) if expected_profit else None,
            'confidence_score': round(confidence_score, 4) if confidence_score else None
        }
        
        return assessment
    
    def batch_assess(self, probabilities: np.ndarray,
                    loan_amounts: np.ndarray = None) -> pd.DataFrame:
        """
        Assess multiple applicants at once
        
        Returns: DataFrame with risk assessments
        """
        assessments = []
        
        for i, prob in enumerate(probabilities):
            loan_amt = loan_amounts[i] if loan_amounts is not None else None
            assessment = self.assess_applicant(prob, loan_amt)
            assessments.append(assessment)
        
        return pd.DataFrame(assessments)
    
    def calculate_portfolio_metrics(self, assessments: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate business metrics for a portfolio of loans
        
        Args:
            assessments: DataFrame from batch_assess()
        
        Returns: Portfolio-level metrics
        """
        total_applicants = len(assessments)
        
        # Decision distribution
        approved = (assessments['decision'] == 'APPROVED').sum()
        review = (assessments['decision'] == 'REVIEW').sum()
        rejected = (assessments['decision'] == 'REJECTED').sum()
        
        # Risk distribution
        low_risk = (assessments['risk_category'] == 'LOW').sum()
        moderate_risk = (assessments['risk_category'] == 'MODERATE').sum()
        high_risk = (assessments['risk_category'] == 'HIGH').sum()
        
        # Average metrics
        avg_default_prob = assessments['default_probability'].mean()
        avg_interest_rate = assessments['interest_rate'].mean()
        
        # Portfolio metrics
        if 'expected_profit' in assessments.columns:
            total_expected_profit = assessments['expected_profit'].sum()
            avg_expected_profit = assessments['expected_profit'].mean()
        else:
            total_expected_profit = None
            avg_expected_profit = None
        
        metrics = {
            'total_applicants': total_applicants,
            'decisions': {
                'approved': int(approved),
                'review': int(review),
                'rejected': int(rejected),
                'approval_rate': round(approved / total_applicants, 4) if total_applicants > 0 else 0
            },
            'risk_distribution': {
                'low': int(low_risk),
                'moderate': int(moderate_risk),
                'high': int(high_risk)
            },
            'averages': {
                'default_probability': round(avg_default_prob, 4),
                'interest_rate': round(avg_interest_rate, 4),
                'expected_profit': round(avg_expected_profit, 2) if avg_expected_profit else None
            },
            'total_expected_profit': round(total_expected_profit, 2) if total_expected_profit else None
        }
        
        return metrics
    
    def calculate_business_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate business cost of predictions
        
        Args:
            y_true: Actual defaults (1) and non-defaults (0)
            y_pred: Predicted defaults (1) and non-defaults (0)
        
        Returns: Total business cost
        """
        from sklearn.metrics import confusion_matrix
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # False Positive cost: Rejected a good loan (lost profit)
        fp_cost = fp * self.business_costs['false_positive_cost']
        
        # False Negative cost: Approved a bad loan (default loss)
        fn_cost = fn * self.business_costs['false_negative_cost']
        
        total_cost = fp_cost + fn_cost
        
        logger.info(f"Business Cost Analysis:")
        logger.info(f"  False Positives: {fp} (Cost: ${fp_cost:,.0f})")
        logger.info(f"  False Negatives: {fn} (Cost: ${fn_cost:,.0f})")
        logger.info(f"  Total Cost: ${total_cost:,.0f}")
        
        return total_cost
    
    def optimize_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                          thresholds: np.ndarray = None) -> Dict[str, Any]:
        """
        Find optimal decision threshold that minimizes business cost
        
        Returns: Optimal threshold and associated metrics
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 50)
        
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = self.calculate_business_cost(y_true, y_pred)
            costs.append(cost)
        
        # Find optimal
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        optimal_cost = costs[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f} (Cost: ${optimal_cost:,.0f})")
        
        return {
            'optimal_threshold': float(optimal_threshold),
            'optimal_cost': float(optimal_cost),
            'all_thresholds': thresholds.tolist(),
            'all_costs': costs
        }


if __name__ == "__main__":
    # Test risk engine
    print("\n=== Risk Engine Test ===\n")
    
    engine = CreditRiskEngine()
    
    # Test individual assessment
    print("Individual Assessment:")
    assessment = engine.assess_applicant(
        default_probability=0.23,
        loan_amount=15000,
        confidence_score=0.94
    )
    for key, value in assessment.items():
        print(f"  {key}: {value}")
    
    # Test batch assessment
    print("\n\nBatch Assessment:")
    probabilities = np.array([0.12, 0.45, 0.78, 0.25, 0.61])
    loan_amounts = np.array([10000, 20000, 15000, 8000, 25000])
    
    assessments = engine.batch_assess(probabilities, loan_amounts)
    print(assessments)
    
    # Portfolio metrics
    print("\n\nPortfolio Metrics:")
    metrics = engine.calculate_portfolio_metrics(assessments)
    import json
    print(json.dumps(metrics, indent=2))
