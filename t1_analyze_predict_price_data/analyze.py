import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load and parse data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Nat_Gas.csv")

df = pd.read_csv(DATA_PATH)
df["Dates"] = pd.to_datetime(df["Dates"], format="%m/%d/%y")
df["Prices"] = df["Prices"].astype(float)
df = df.sort_values("Dates").reset_index(drop=True)

# Numeric representation: fractional years from the first data point
origin = df["Dates"].iloc[0]
df["t"] = (df["Dates"] - origin).dt.days / 365.25

print("=== Natural Gas Price Data ===")
print(f"Date range : {df['Dates'].min().date()} to {df['Dates'].max().date()}")
print(f"Price range: ${df['Prices'].min():.2f} – ${df['Prices'].max():.2f}")
print(f"Data points: {len(df)}\n")

# ---------------------------------------------------------------------------
# 2. Visualise raw data
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Natural Gas Price Analysis", fontsize=15, fontweight="bold")

ax = axes[0, 0]
ax.plot(df["Dates"], df["Prices"], "o-", color="steelblue", markersize=4)
ax.set_title("Monthly End-of-Month Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.grid(True, alpha=0.3)

# Seasonal box-plot by month
ax = axes[0, 1]
df["Month"] = df["Dates"].dt.month
monthly_groups = [group["Prices"].values for _, group in df.groupby("Month")]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax.boxplot(monthly_groups, labels=month_labels)
ax.set_title("Seasonal Distribution by Month")
ax.set_ylabel("Price ($)")
ax.grid(True, alpha=0.3, axis="y")

# ---------------------------------------------------------------------------
# 3. Model: linear trend + sinusoidal seasonality
# ---------------------------------------------------------------------------
# Price(t) = a*t + b + A*sin(2π*t + φ₁) + B*sin(4π*t + φ₂)
#
# - Linear term captures the multi-year upward drift
# - First sine (period = 1 year) captures the dominant annual cycle
# - Second sine (period = 6 months) captures the secondary harmonic
# ---------------------------------------------------------------------------


def price_model(t, a, b, A, phi1, B, phi2):
    trend = a * t + b
    season_annual = A * np.sin(2 * np.pi * t + phi1)
    season_semi = B * np.sin(4 * np.pi * t + phi2)
    return trend + season_annual + season_semi


t_data = df["t"].values
p_data = df["Prices"].values

p0 = [0.5, 10.5, 0.5, 0.0, 0.2, 0.0]
params, cov = curve_fit(price_model, t_data, p_data, p0=p0, maxfev=10000)
a, b, A, phi1, B, phi2 = params

print("=== Fitted Model Parameters ===")
print(f"Linear trend : slope (a) = {a:.4f} $/year,  intercept (b) = {b:.4f}")
print(f"Annual cycle : amplitude (A) = {abs(A):.4f},  phase (φ₁) = {phi1:.4f} rad")
print(f"Semi-annual  : amplitude (B) = {abs(B):.4f},  phase (φ₂) = {phi2:.4f} rad")

residuals = p_data - price_model(t_data, *params)
rmse = np.sqrt(np.mean(residuals ** 2))
print(f"RMSE         : ${rmse:.4f}\n")

# ---------------------------------------------------------------------------
# 4. Estimation function — the deliverable
# ---------------------------------------------------------------------------


def estimate_price(date_str: str) -> float:
    """Return estimated natural gas price for any date (past or future).

    Parameters
    ----------
    date_str : str
        Date in 'MM/DD/YYYY' or any pandas-parseable format.

    Returns
    -------
    float
        Estimated price in dollars.
    """
    date = pd.to_datetime(date_str)
    t = (date - origin).days / 365.25
    return float(price_model(t, *params))


# ---------------------------------------------------------------------------
# 5. Plot fitted curve and one-year extrapolation
# ---------------------------------------------------------------------------

last_date = df["Dates"].max()
future_end = last_date + timedelta(days=365)
date_range = pd.date_range(start=df["Dates"].min(), end=future_end, freq="D")
t_range = (date_range - origin).days / 365.25

predicted = price_model(t_range, *params)

ax = axes[1, 0]
ax.plot(df["Dates"], df["Prices"], "o", color="steelblue", markersize=4,
        label="Observed data")
ax.plot(date_range, predicted, "-", color="coral", linewidth=1.5,
        label="Fitted model")
ax.axvline(last_date, color="gray", linestyle="--", alpha=0.6,
           label="Extrapolation starts")
ax.set_title("Model Fit + 1-Year Extrapolation")
ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Residuals
ax = axes[1, 1]
ax.bar(df["Dates"], residuals, width=20, color="mediumseagreen", alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title(f"Residuals (RMSE = ${rmse:.3f})")
ax.set_xlabel("Date")
ax.set_ylabel("Residual ($)")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "price_analysis.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# 6. Print sample estimates
# ---------------------------------------------------------------------------

print("=== Sample Price Estimates ===")
sample_dates = [
    "10/31/2020", "06/30/2022", "12/31/2023",   # within data range
    "12/31/2024", "03/31/2025", "09/30/2025",    # extrapolated
]
for d in sample_dates:
    tag = "observed" if pd.to_datetime(d) <= last_date else "extrapolated"
    print(f"  {d:12s} -> ${estimate_price(d):6.2f}  ({tag})")

print("\n=== Observations on Seasonal Patterns ===")
print(f"""
Natural gas prices exhibit a clear annual cycle driven by:
 • Heating demand  – prices peak in winter (Dec-Feb) when residential
   and commercial heating consumption surges.
 • Cooling demand  – a secondary, smaller rise in summer (Jul-Aug) from
   gas-fired electricity generation for air conditioning.
 • Shoulder months – spring (Apr-May) and autumn (Sep-Oct) see the
   lowest prices as neither heating nor cooling demand dominates.

The data also shows a modest upward linear trend (~${a:.2f}/year),
reflecting gradual supply/demand shifts over the 2020-2024 window.

The model uses a linear trend plus two sinusoidal harmonics (annual
and semi-annual) to capture these patterns, giving an RMSE of ${rmse:.3f}.
""")

# ---------------------------------------------------------------------------
# Interactive usage:  estimate_price("MM/DD/YYYY") -> float
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query_date = sys.argv[1]
        price = estimate_price(query_date)
        print(f"\nEstimated price on {query_date}: ${price:.2f}")
