import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

RECOVERY_RATE = 0.10
DATA_PATH = "Task 3 and 4_Loan_Data.csv"


def load_and_explore_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nBasic statistics:\n{df.describe()}")
    print(f"\nDefault rate: {df['default'].mean():.4f} ({df['default'].sum()} / {len(df)})")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


def engineer_features(df):
    """Create additional features that may improve prediction."""
    df = df.copy()
    df['debt_to_income'] = df['total_debt_outstanding'] / df['income']
    df['loan_to_income'] = df['loan_amt_outstanding'] / df['income']
    df['debt_per_credit_line'] = np.where(
        df['credit_lines_outstanding'] > 0,
        df['total_debt_outstanding'] / df['credit_lines_outstanding'],
        df['total_debt_outstanding']
    )
    return df


def prepare_data(df):
    """Split into features/target and train/test sets."""
    feature_cols = [
        'credit_lines_outstanding', 'loan_amt_outstanding',
        'total_debt_outstanding', 'income', 'years_employed', 'fico_score',
        'debt_to_income', 'loan_to_income', 'debt_per_credit_line'
    ]
    X = df[feature_cols]
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def train_models(X_train_scaled, X_train, y_train):
    """Train multiple models for comparative analysis."""
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, max_depth=5, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, max_depth=10, class_weight='balanced'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, random_state=42, max_depth=5, learning_rate=0.1
        ),
    }

    trained = {}
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        trained[name] = model

    return trained


def evaluate_models(trained_models, X_test_scaled, X_test, y_test):
    """Evaluate all models and return metrics."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results = {}
    for name, model in trained_models.items():
        X_eval = X_test_scaled if name == 'Logistic Regression' else X_test
        y_prob = model.predict_proba(X_eval)[:, 1]
        y_pred = model.predict(X_eval)
        auc = roc_auc_score(y_test, y_prob)

        cv_X = X_test_scaled if name == 'Logistic Regression' else X_test
        cv_scores = cross_val_score(model, cv_X, y_test, cv=5, scoring='roc_auc')

        results[name] = {
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_prob': y_prob,
            'y_pred': y_pred,
        }
        print(f"\n{name}:")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  CV AUC:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Classification Report:")
        report = classification_report(y_test, y_pred)
        for line in report.split('\n'):
            print(f"    {line}")

    return results


def plot_results(results, y_test, trained_models, feature_cols, X_train):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ROC curves
    ax = axes[0, 0]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Feature importance (from Random Forest)
    ax = axes[0, 1]
    rf_model = trained_models['Random Forest']
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)
    ax.barh(range(len(sorted_idx)), importances[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_title('Feature Importance (Random Forest)')
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.3)

    # Probability distribution for best model
    ax = axes[1, 0]
    best_name = max(results, key=lambda k: results[k]['auc'])
    best_probs = results[best_name]['y_prob']
    ax.hist(best_probs[y_test == 0], bins=50, alpha=0.6, label='Non-Default', density=True)
    ax.hist(best_probs[y_test == 1], bins=50, alpha=0.6, label='Default', density=True)
    ax.set_xlabel('Predicted Probability of Default')
    ax.set_ylabel('Density')
    ax.set_title(f'PD Distribution ({best_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Model comparison bar chart
    ax = axes[1, 1]
    names = list(results.keys())
    aucs = [results[n]['auc'] for n in names]
    colors = ['#2ecc71' if a == max(aucs) else '#3498db' for a in aucs]
    bars = ax.bar(names, aucs, color=colors)
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Model AUC Comparison')
    ax.set_ylim(min(aucs) - 0.05, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{auc:.4f}', ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig('credit_risk_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nPlot saved to credit_risk_analysis.png")


def calculate_expected_loss(probability_of_default, loan_amount, recovery_rate=RECOVERY_RATE):
    """
    Calculate expected loss on a loan.

    Expected Loss = PD × LGD × EAD
    where LGD = 1 - Recovery Rate, EAD = loan amount outstanding
    """
    lgd = 1 - recovery_rate
    return probability_of_default * lgd * loan_amount


class CreditRiskModel:
    """Production-ready credit risk model for predicting PD and expected loss."""

    def __init__(self, model, scaler, feature_cols, recovery_rate=RECOVERY_RATE):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.recovery_rate = recovery_rate
        self.requires_scaling = isinstance(model, LogisticRegression)

    def _prepare_input(self, loan_properties):
        """Convert input dict to model-ready DataFrame."""
        df = pd.DataFrame([loan_properties])
        df['debt_to_income'] = df['total_debt_outstanding'] / df['income']
        df['loan_to_income'] = df['loan_amt_outstanding'] / df['income']
        df['debt_per_credit_line'] = np.where(
            df['credit_lines_outstanding'] > 0,
            df['total_debt_outstanding'] / df['credit_lines_outstanding'],
            df['total_debt_outstanding']
        )
        return df[self.feature_cols]

    def predict_probability_of_default(self, loan_properties):
        """
        Predict the probability of default for a borrower.

        Parameters:
            loan_properties: dict with keys:
                - credit_lines_outstanding (int)
                - loan_amt_outstanding (float)
                - total_debt_outstanding (float)
                - income (float)
                - years_employed (int)
                - fico_score (int)

        Returns:
            float: probability of default (0 to 1)
        """
        X = self._prepare_input(loan_properties)
        if self.requires_scaling:
            X = pd.DataFrame(self.scaler.transform(X), columns=self.feature_cols)
        return float(self.model.predict_proba(X)[:, 1][0])

    def predict_expected_loss(self, loan_properties):
        """
        Predict the expected loss on a loan.

        Expected Loss = PD × (1 - Recovery Rate) × Loan Amount

        Parameters:
            loan_properties: dict with keys as in predict_probability_of_default

        Returns:
            dict with pd, lgd, ead, expected_loss
        """
        pd_value = self.predict_probability_of_default(loan_properties)
        loan_amount = loan_properties['loan_amt_outstanding']
        lgd = 1 - self.recovery_rate
        expected_loss = pd_value * lgd * loan_amount

        return {
            'probability_of_default': pd_value,
            'loss_given_default': lgd,
            'exposure_at_default': loan_amount,
            'expected_loss': expected_loss,
            'recovery_rate': self.recovery_rate,
        }


def main():
    # Load and explore
    df = load_and_explore_data()

    # Feature engineering
    df = engineer_features(df)

    # Prepare data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = prepare_data(df)

    # Train models
    trained_models = train_models(X_train_scaled, X_train, y_train)

    # Evaluate
    results = evaluate_models(trained_models, X_test_scaled, X_test, y_test)

    # Plot
    plot_results(results, y_test, trained_models, feature_cols, X_train)

    # Select best model
    best_name = max(results, key=lambda k: results[k]['auc'])
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (AUC = {results[best_name]['auc']:.4f})")
    print(f"{'=' * 60}")

    best_model = trained_models[best_name]
    risk_model = CreditRiskModel(best_model, scaler, feature_cols)

    # Demo predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    examples = [
        {
            'description': 'Low-risk borrower (high income, high FICO, low debt)',
            'credit_lines_outstanding': 0,
            'loan_amt_outstanding': 5000,
            'total_debt_outstanding': 2000,
            'income': 90000,
            'years_employed': 8,
            'fico_score': 750,
        },
        {
            'description': 'Medium-risk borrower',
            'credit_lines_outstanding': 2,
            'loan_amt_outstanding': 4000,
            'total_debt_outstanding': 8000,
            'income': 50000,
            'years_employed': 3,
            'fico_score': 620,
        },
        {
            'description': 'High-risk borrower (low income, low FICO, high debt)',
            'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 3000,
            'total_debt_outstanding': 15000,
            'income': 25000,
            'years_employed': 1,
            'fico_score': 520,
        },
    ]

    for ex in examples:
        desc = ex.pop('description')
        result = risk_model.predict_expected_loss(ex)
        print(f"\n{desc}:")
        print(f"  Probability of Default: {result['probability_of_default']:.4f} "
              f"({result['probability_of_default']*100:.2f}%)")
        print(f"  Loss Given Default:     {result['loss_given_default']:.2f} "
              f"(Recovery Rate: {result['recovery_rate']:.0%})")
        print(f"  Exposure at Default:    ${result['exposure_at_default']:,.2f}")
        print(f"  Expected Loss:          ${result['expected_loss']:,.2f}")

    # Portfolio-level analysis
    print("\n" + "=" * 60)
    print("PORTFOLIO-LEVEL EXPECTED LOSS")
    print("=" * 60)

    test_df = df.iloc[X_test.index].copy()
    X_eval = X_test_scaled if risk_model.requires_scaling else X_test
    test_df['predicted_pd'] = best_model.predict_proba(X_eval)[:, 1]
    test_df['expected_loss'] = (
        test_df['predicted_pd'] * (1 - RECOVERY_RATE) * test_df['loan_amt_outstanding']
    )

    total_exposure = test_df['loan_amt_outstanding'].sum()
    total_expected_loss = test_df['expected_loss'].sum()
    avg_pd = test_df['predicted_pd'].mean()

    print(f"  Total Exposure (test set):    ${total_exposure:,.2f}")
    print(f"  Total Expected Loss:          ${total_expected_loss:,.2f}")
    print(f"  Loss as % of Exposure:        {total_expected_loss/total_exposure*100:.2f}%")
    print(f"  Average PD across portfolio:  {avg_pd:.4f} ({avg_pd*100:.2f}%)")

    return risk_model


if __name__ == '__main__':
    model = main()
