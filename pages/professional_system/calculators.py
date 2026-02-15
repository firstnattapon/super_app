import numpy as np
import math
from typing import Tuple

def norm_cdf(x):
    """Calculates Normal CDF using math.erf (No scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Tuple[float, float]:
    """
    Calculates Option Price and Delta.
    Returns: (Price, Delta)
    """
    if T <= 0:
        val = max(S - K, 0) if option_type.lower() == 'call' else max(K - S, 0)
        return val, 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
    else:
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1

    return price, delta

def calculate_rebalance(current_price: float, fix_capital: float, shares_held: float) -> Tuple[str, float, float]:
    """
    Determines if we need to Buy or Sell to return to Fix Capital.
    Returns: (Action, Share_Diff, Cash_Diff)
    """
    current_val = shares_held * current_price
    diff = current_val - fix_capital

    if abs(diff) < 1.0: # Threshold
        return "Hold", 0.0, 0.0

    if diff > 0:
        # Sell excess
        shares_to_sell = diff / current_price
        return "Sell", shares_to_sell, diff
    else:
        # Buy deficit
        shares_to_buy = abs(diff) / current_price
        return "Buy", shares_to_buy, abs(diff)

def calculate_shannon_benchmark(fix_c: float, p0: float, pt: float) -> float:
    """Calculates theoretical Shannon's Demon value: fix_c * ln(Pt/P0)"""
    if p0 == 0 or pt == 0: return 0.0
    return fix_c * math.log(pt / p0)
