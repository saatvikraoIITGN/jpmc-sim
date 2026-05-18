import numpy as np
from datetime import datetime

# Pre-fitted model parameters (from analyze.py curve_fit)
# Price(t) = a*t + b + A*sin(2π*t + φ₁) + B*sin(4π*t + φ₂)
# where t = years since Oct 31, 2020
ORIGIN = datetime(2020, 10, 31)
a = 0.543243564189369
b = 10.1409192060573
A = 0.6869887518394823
phi1 = -0.0385204820078225
B = -0.09127969821664632
phi2 = 1.0648326565090214


def estimate_price(date_str: str) -> float:
    """Return estimated natural gas price for any date (past or future).

    Parameters
    ----------
    date_str : str
        Date in 'MM/DD/YYYY' format.

    Returns
    -------
    float
        Estimated price in dollars.
    """
    date = datetime.strptime(date_str, "%m/%d/%Y")
    t = (date - ORIGIN).days / 365.25
    price = a * t + b + A * np.sin(2 * np.pi * t + phi1) + B * np.sin(4 * np.pi * t + phi2)
    return float(price)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query_date = sys.argv[1]
        price = estimate_price(query_date)
        print(f"Estimated price on {query_date}: ${price:.2f}")
    else:
        print("Usage: python predict.py <date>")
