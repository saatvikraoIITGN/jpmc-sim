"""
Commodity Storage Contract Pricing Model

Prices a natural gas storage contract given injection/withdrawal schedules,
storage constraints, and associated costs. Uses the price estimation model
from Task 1 to look up buy/sell prices on specified dates.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'task1'))
from predict import estimate_price


def price_contract(
    injection_dates: List[str],
    withdrawal_dates: List[str],
    injection_prices: Optional[List[float]] = None,
    withdrawal_prices: Optional[List[float]] = None,
    injection_volumes: Optional[List[float]] = None,
    withdrawal_volumes: Optional[List[float]] = None,
    injection_rate: float = 1_000_000,
    withdrawal_rate: float = 1_000_000,
    max_storage_volume: float = 5_000_000,
    storage_cost_per_month: float = 100_000,
    injection_withdrawal_cost_per_mmbtu: float = 0.01,
    transport_cost_per_trip: float = 50_000,
) -> dict:
    """Price a natural gas storage contract.

    Simulates all cash flows for a buy-store-sell strategy: purchases on
    injection dates, sales on withdrawal dates, monthly storage rental,
    injection/withdrawal handling fees, and transport costs.

    Parameters
    ----------
    injection_dates : list of str
        Dates to buy gas and inject into storage ('MM/DD/YYYY').
    withdrawal_dates : list of str
        Dates to withdraw gas and sell ('MM/DD/YYYY').
    injection_prices : list of float, optional
        Purchase price ($/MMBtu) on each injection date.
        If None, estimated from the pricing model.
    withdrawal_prices : list of float, optional
        Sale price ($/MMBtu) on each withdrawal date.
        If None, estimated from the pricing model.
    injection_volumes : list of float, optional
        Volume (MMBtu) to inject on each date. Defaults to `injection_rate`
        for every injection date.
    withdrawal_volumes : list of float, optional
        Volume (MMBtu) to withdraw on each date. Defaults to
        `withdrawal_rate` for every withdrawal date.
    injection_rate : float
        Default volume injected per injection event (MMBtu).
    withdrawal_rate : float
        Default volume withdrawn per withdrawal event (MMBtu).
    max_storage_volume : float
        Maximum storage capacity (MMBtu).
    storage_cost_per_month : float
        Fixed monthly rental fee ($) for the storage facility.
    injection_withdrawal_cost_per_mmbtu : float
        Handling fee ($/MMBtu) charged on each injection or withdrawal.
    transport_cost_per_trip : float
        Flat transport cost ($) per injection or withdrawal trip.

    Returns
    -------
    dict
        Breakdown of revenues, costs, and net contract value.
    """
    if injection_prices is None:
        injection_prices = [estimate_price(d) for d in injection_dates]
    if withdrawal_prices is None:
        withdrawal_prices = [estimate_price(d) for d in withdrawal_dates]
    if injection_volumes is None:
        injection_volumes = [injection_rate] * len(injection_dates)
    if withdrawal_volumes is None:
        withdrawal_volumes = [withdrawal_rate] * len(withdrawal_dates)

    if len(injection_dates) != len(injection_prices):
        raise ValueError("injection_dates and injection_prices must have the same length")
    if len(withdrawal_dates) != len(withdrawal_prices):
        raise ValueError("withdrawal_dates and withdrawal_prices must have the same length")
    if len(injection_dates) != len(injection_volumes):
        raise ValueError("injection_dates and injection_volumes must have the same length")
    if len(withdrawal_dates) != len(withdrawal_volumes):
        raise ValueError("withdrawal_dates and withdrawal_volumes must have the same length")

    # --- simulate storage volume over time & compute cash flows ---
    purchase_cost = 0.0
    sale_revenue = 0.0
    total_injection_volume = 0.0
    total_withdrawal_volume = 0.0
    stored_volume = 0.0

    events = []
    for i, d in enumerate(injection_dates):
        events.append(("inject", datetime.strptime(d, "%m/%d/%Y"), injection_prices[i], injection_volumes[i]))
    for i, d in enumerate(withdrawal_dates):
        events.append(("withdraw", datetime.strptime(d, "%m/%d/%Y"), withdrawal_prices[i], withdrawal_volumes[i]))

    events.sort(key=lambda x: x[1])

    for event_type, date, price, volume in events:
        if event_type == "inject":
            if stored_volume + volume > max_storage_volume:
                actual = max_storage_volume - stored_volume
                if actual <= 0:
                    continue
                volume = actual
            stored_volume += volume
            purchase_cost += price * volume
            total_injection_volume += volume
        else:
            if volume > stored_volume:
                volume = stored_volume
                if volume <= 0:
                    continue
            stored_volume -= volume
            sale_revenue += price * volume
            total_withdrawal_volume += volume

    # --- storage rental: monthly fee for the span the facility is rented ---
    all_dates = [datetime.strptime(d, "%m/%d/%Y") for d in injection_dates + withdrawal_dates]
    start_date = min(all_dates)
    end_date = max(all_dates)
    months_rented = ((end_date.year - start_date.year) * 12
                     + (end_date.month - start_date.month))
    months_rented = max(months_rented, 1)
    total_storage_cost = storage_cost_per_month * months_rented

    # --- injection / withdrawal handling fees ---
    total_handling_volume = total_injection_volume + total_withdrawal_volume
    handling_cost = injection_withdrawal_cost_per_mmbtu * total_handling_volume

    # --- transport: one trip per injection event + one per withdrawal event ---
    num_trips = len(injection_dates) + len(withdrawal_dates)
    total_transport_cost = transport_cost_per_trip * num_trips

    # --- net value ---
    total_costs = purchase_cost + total_storage_cost + handling_cost + total_transport_cost
    net_value = sale_revenue - total_costs

    return {
        "sale_revenue": sale_revenue,
        "purchase_cost": purchase_cost,
        "storage_cost": total_storage_cost,
        "handling_cost": handling_cost,
        "transport_cost": total_transport_cost,
        "total_costs": total_costs,
        "net_value": net_value,
        "months_rented": months_rented,
        "total_injected_mmbtu": total_injection_volume,
        "total_withdrawn_mmbtu": total_withdrawal_volume,
    }


def print_result(result: dict, title: str = "Contract Valuation") -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Sale revenue:          ${result['sale_revenue']:>15,.2f}")
    print(f"  Purchase cost:        -${result['purchase_cost']:>15,.2f}")
    print(f"  Storage cost:         -${result['storage_cost']:>15,.2f}")
    print(f"  Handling cost:        -${result['handling_cost']:>15,.2f}")
    print(f"  Transport cost:       -${result['transport_cost']:>15,.2f}")
    print(f"  {'─'*51}")
    print(f"  NET CONTRACT VALUE:    ${result['net_value']:>15,.2f}")
    print(f"{'='*55}")
    print(f"  Storage period:         {result['months_rented']} months")
    print(f"  Total injected:         {result['total_injected_mmbtu']:,.0f} MMBtu")
    print(f"  Total withdrawn:        {result['total_withdrawn_mmbtu']:,.0f} MMBtu")
    print()


# ---------------------------------------------------------------------------
# Sample test cases
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # --- Test 1: Textbook example from the task description ---
    print("\n▸ Test 1 – Textbook example (buy summer @ $2, sell winter @ $3)")
    result1 = price_contract(
        injection_dates=["06/01/2023"],
        withdrawal_dates=["10/01/2023"],
        injection_prices=[2.0],
        withdrawal_prices=[3.0],
        injection_volumes=[1_000_000],
        withdrawal_volumes=[1_000_000],
        max_storage_volume=5_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_mmbtu=0.005,
        transport_cost_per_trip=50_000,
    )
    print_result(result1, "Textbook Example")

    # --- Test 2: Model-estimated prices, single buy/sell ---
    print("▸ Test 2 – Buy summer 2024, sell winter 2024 (model prices)")
    result2 = price_contract(
        injection_dates=["06/30/2024"],
        withdrawal_dates=["12/31/2024"],
        injection_volumes=[1_000_000],
        withdrawal_volumes=[1_000_000],
        max_storage_volume=5_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_mmbtu=0.01,
        transport_cost_per_trip=50_000,
    )
    print_result(result2, "Summer→Winter 2024 (model prices)")

    # --- Test 3: Multiple injection and withdrawal dates ---
    print("▸ Test 3 – Multiple injections (Apr–Jun 2025), withdrawals (Nov 2025–Jan 2026)")
    result3 = price_contract(
        injection_dates=["04/30/2025", "05/31/2025", "06/30/2025"],
        withdrawal_dates=["11/30/2025", "12/31/2025", "01/31/2026"],
        injection_volumes=[500_000, 500_000, 500_000],
        withdrawal_volumes=[500_000, 500_000, 500_000],
        max_storage_volume=2_000_000,
        storage_cost_per_month=120_000,
        injection_withdrawal_cost_per_mmbtu=0.01,
        transport_cost_per_trip=50_000,
    )
    print_result(result3, "Multi-date Strategy 2025–2026")

    # --- Test 4: Storage capacity constraint ---
    print("▸ Test 4 – Storage cap hit (try to inject 6M into 5M facility)")
    result4 = price_contract(
        injection_dates=["05/31/2024", "06/30/2024"],
        withdrawal_dates=["01/31/2025"],
        injection_prices=[10.0, 10.0],
        withdrawal_prices=[14.0],
        injection_volumes=[3_000_000, 3_000_000],
        withdrawal_volumes=[5_000_000],
        max_storage_volume=5_000_000,
        storage_cost_per_month=100_000,
        injection_withdrawal_cost_per_mmbtu=0.01,
        transport_cost_per_trip=50_000,
    )
    print_result(result4, "Storage Cap Constraint")
