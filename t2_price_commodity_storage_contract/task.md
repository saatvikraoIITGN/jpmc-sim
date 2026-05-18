# Task 2: Price a commodity storage contract

#### Here is the background information on your task: 
Great work! The desk now has the price data they need. The final ingredient before they can begin trading with the client is the pricing model. Alex tells you the client wants to start trading as soon as possible. They believe the winter will be colder than expected, so they want to buy gas now to store and sell in winter in order to take advantage of the resulting increase in gas prices. They ask you to write a script that they can use to price the contract. Once the desk are happy, you will work with engineering, risk, and model validation to incorporate this model into production code.

The concept is simple: any trade agreement is as valuable as the price you can sell minus the price at which you are able to buy. Any cost incurred as part of executing this agreement is also deducted from the overall value. So, for example, if I can purchase a million MMBtu of natural gas in summer at $2/MMBtu, store this for four months, and ensure that I can sell the same quantity at $3/MMBtu without incurring any additional costs, the value of this contract would be ($3-$2) *1e6 = $1million. If there are costs involved, such as having to pay the storage facility owner a fixed fee of $100K a month, then the 'value' of the contract, from my perspective, would drop by the overall rental amount to $600K. Another cost could be the injection/withdrawal cost, like having to pay the storage facility owner $10K per 1 million MMBtu for injection/withdrawal, then the price will further go down by $10K to $590K. Additionally, if I am supposed to foot a bill of $50K each time for transporting the gas to and from the facility, the cost of this contract would fall by another $100K. Think of the valuation as a fair estimate at which both the trading desk and the client would be happy to enter into the contract. 


#### Here is your task: 
You need to create a prototype pricing model that can go through further validation and testing before being put into production. Eventually, this model may be the basis for fully automated quoting to clients, but for now, the desk will use it with manual oversight to explore options with the client. 

You should write a function that is able to use the data you created previously to price the contract. The client may want to choose multiple dates to inject and withdraw a set amount of gas, so your approach should generalize the explanation from before. Consider all the cash flows involved in the product.

The input parameters that should be taken into account for pricing are:

- Injection dates. 
- Withdrawal dates.
- The prices at which the commodity can be purchased/sold on those dates.
- The rate at which the gas can be injected/withdrawn.
- The maximum volume that can be stored.
- Storage costs.

Write a function that takes these inputs and gives back the value of the contract. You can assume there is no transport delay and that interest rates are zero. Market holidays, weekends, and bank holidays need not be accounted for. Test your code by selecting a few sample inputs.

---

## Solution

### Approach

The pricing model is implemented in `price_contract.py`. The core idea is:

**Net Contract Value = Sale Revenue − Purchase Cost − Storage Cost − Handling Cost − Transport Cost**

The function `price_contract()` generalizes this to any number of injection and withdrawal dates. It processes events chronologically, tracking the volume held in storage at every step and enforcing the facility's maximum capacity.

### Cash flow components

| Component | Calculation |
|---|---|
| **Purchase cost** | `buy_price × volume` on each injection date |
| **Sale revenue** | `sell_price × volume` on each withdrawal date |
| **Storage rental** | `monthly_fee × months` spanning the first to last event |
| **Handling fees** | `cost_per_MMBtu × (total_injected + total_withdrawn)` |
| **Transport** | `cost_per_trip × number_of_trips` (one trip per injection + one per withdrawal) |

### Function signature

```python
def price_contract(
    injection_dates,          # List of buy dates ('MM/DD/YYYY')
    withdrawal_dates,         # List of sell dates ('MM/DD/YYYY')
    injection_prices=None,    # $/MMBtu on each buy date (auto-estimated if omitted)
    withdrawal_prices=None,   # $/MMBtu on each sell date (auto-estimated if omitted)
    injection_volumes=None,   # MMBtu to inject each date
    withdrawal_volumes=None,  # MMBtu to withdraw each date
    injection_rate,           # Default injection volume per event
    withdrawal_rate,          # Default withdrawal volume per event
    max_storage_volume,       # Facility capacity (MMBtu)
    storage_cost_per_month,   # Fixed monthly rental ($)
    injection_withdrawal_cost_per_mmbtu,  # Handling fee ($/MMBtu)
    transport_cost_per_trip,  # Flat fee ($) per trip
) -> dict
```

If prices are not provided, the function automatically estimates them using the sinusoidal regression model built in Task 1 (`predict.py`).

### Key design decisions

1. **Chronological event simulation** — Injection and withdrawal events are merged into a single timeline sorted by date. This correctly handles interleaved buy/sell schedules and keeps the stored-volume ledger accurate.
2. **Storage capacity enforcement** — If an injection would exceed `max_storage_volume`, only the remaining headroom is filled. Similarly, a withdrawal cannot exceed the current stored volume.
3. **Flexible pricing** — Prices can be supplied explicitly (e.g., from forward curves or broker quotes) or left as `None` to use the model's estimates. This lets the desk quickly explore scenarios.

### Test results

Four sample scenarios were run to validate the model:

**Test 1 — Textbook example** (buy 1M MMBtu @ $2, sell @ $3, 4 months storage, $100K/month rent, $0.005/MMBtu handling, $50K/trip transport):

| Line item | Amount |
|---|---|
| Sale revenue | $3,000,000 |
| Purchase cost | −$2,000,000 |
| Storage cost (4 months) | −$400,000 |
| Handling cost | −$10,000 |
| Transport cost (2 trips) | −$100,000 |
| **Net value** | **$490,000** |

**Test 2 — Summer → Winter 2024** (model-estimated prices, 1M MMBtu, 6-month hold, $100K/month rent, $0.01/MMBtu handling, $50K/trip transport):

| Line item | Amount |
|---|---|
| Sale revenue | $12,988,688 |
| Purchase cost | −$11,555,365 |
| Storage cost (6 months) | −$600,000 |
| Handling cost | −$20,000 |
| Transport cost (2 trips) | −$100,000 |
| **Net value** | **$713,324** |

**Test 3 — Multi-date strategy 2025–2026** (3 injections Apr–Jun, 3 withdrawals Nov–Jan, 500K MMBtu each, 9-month span, $120K/month rent, $0.01/MMBtu handling, $50K/trip transport):

| Line item | Amount |
|---|---|
| Sale revenue | $20,212,568 |
| Purchase cost | −$18,445,355 |
| Storage cost (9 months) | −$1,080,000 |
| Handling cost | −$30,000 |
| Transport cost (6 trips) | −$300,000 |
| **Net value** | **$357,214** |

**Test 4 — Storage capacity constraint** (attempt to inject 6M MMBtu into a 5M facility, 8-month span, $100K/month rent, $0.01/MMBtu handling, $50K/trip transport):

| Line item | Amount |
|---|---|
| Sale revenue | $70,000,000 |
| Purchase cost | −$50,000,000 |
| Storage cost (8 months) | −$800,000 |
| Handling cost | −$100,000 |
| Transport cost (3 trips) | −$150,000 |
| **Net value** | **$18,950,000** |

### Assumptions

- No transport delay.
- Interest rates are zero (no discounting of future cash flows).
- Market holidays, weekends, and bank holidays are not modelled.
- Storage rental is charged for entire calendar months from first event to last event.
- One transport trip is required per injection or withdrawal event.
