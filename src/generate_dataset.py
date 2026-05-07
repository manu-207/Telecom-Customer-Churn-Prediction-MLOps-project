"""
Synthetic dataset generator for the MLOps champion/challenger project.
Generates a realistic binary classification dataset simulating
customer churn prediction.
"""

import os
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer churn dataset.

    Features:
        - tenure: months as customer (0-72)
        - monthly_charges: monthly bill amount
        - total_charges: cumulative charges
        - contract_type: Month-to-month, One year, Two year
        - payment_method: Electronic check, Mailed check, Bank transfer, Credit card
        - internet_service: DSL, Fiber optic, No
        - online_security: Yes, No
        - tech_support: Yes, No
        - num_support_tickets: support tickets filed (0-15)
        - avg_monthly_usage_gb: average data usage
        - late_payments: number of late payments
        - age: customer age
    Target:
        - churn: 0 (stayed) or 1 (churned)
    """
    rng = np.random.RandomState(random_state)

    # Numeric features
    tenure = rng.randint(0, 73, n_samples)
    monthly_charges = rng.uniform(20, 120, n_samples).round(2)
    total_charges = (tenure * monthly_charges + rng.normal(0, 50, n_samples)).clip(0).round(2)
    num_support_tickets = rng.poisson(2, n_samples).clip(0, 15)
    avg_monthly_usage_gb = rng.exponential(50, n_samples).round(1)
    late_payments = rng.poisson(1, n_samples).clip(0, 10)
    age = rng.randint(18, 80, n_samples)

    # Categorical features
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.25, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.35, 0.45, 0.20]
    )
    online_security = rng.choice(["Yes", "No"], n_samples, p=[0.4, 0.6])
    tech_support = rng.choice(["Yes", "No"], n_samples, p=[0.35, 0.65])

    # Build churn probability based on realistic patterns
    churn_prob = np.zeros(n_samples)

    # Short tenure increases churn
    churn_prob += np.where(tenure < 12, 0.15, 0.0)
    churn_prob += np.where(tenure < 6, 0.10, 0.0)

    # Month-to-month contracts churn more
    churn_prob += np.where(contract_type == "Month-to-month", 0.20, 0.0)
    churn_prob += np.where(contract_type == "Two year", -0.10, 0.0)

    # High monthly charges increase churn
    churn_prob += np.where(monthly_charges > 80, 0.10, 0.0)
    churn_prob += np.where(monthly_charges > 100, 0.05, 0.0)

    # Fiber optic with no security/support churns more
    fiber_no_security = (internet_service == "Fiber optic") & (online_security == "No")
    churn_prob += np.where(fiber_no_security, 0.12, 0.0)

    # Many support tickets = frustrated customer
    churn_prob += np.where(num_support_tickets > 4, 0.10, 0.0)

    # Late payments correlate with churn
    churn_prob += np.where(late_payments > 2, 0.08, 0.0)

    # No tech support increases churn
    churn_prob += np.where(tech_support == "No", 0.05, 0.0)

    # Electronic check payment method correlates with churn
    churn_prob += np.where(payment_method == "Electronic check", 0.08, 0.0)

    # Normalize to valid probability range
    churn_prob = churn_prob.clip(0.05, 0.85)

    # Add noise
    churn_prob += rng.normal(0, 0.05, n_samples)
    churn_prob = churn_prob.clip(0.0, 1.0)

    # Generate target
    churn = (rng.random(n_samples) < churn_prob).astype(int)

    # Assemble DataFrame
    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "num_support_tickets": num_support_tickets,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "late_payments": late_payments,
        "age": age,
        "churn": churn,
    })

    return df


def main():
    """Generate and save the dataset."""
    print("Generating synthetic churn dataset...")
    df = generate_churn_dataset(n_samples=5000)

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/dataset.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    print(f"\nFeature types:")
    print(f"  Numeric:     {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  Categorical: {df.select_dtypes(include=['object']).shape[1]}")
    print(f"\nSample rows:")
    print(df.head())


if __name__ == "__main__":
    main()
