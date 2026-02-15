from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import date

@dataclass
class OptionPosition:
    """Represents an Option contract (LEAPS, Put, Short Call/Put)."""
    symbol: str
    option_type: str  # 'Call' or 'Put'
    strike: float
    expiration: date
    qty: int  # Number of contracts
    premium_paid: float  # Per share
    current_price: float = 0.0
    strategy_type: str = "LEAPS"  # LEAPS, Shield, Income

    @property
    def cost_basis(self) -> float:
        return self.qty * 100 * self.premium_paid

    @property
    def current_value(self) -> float:
        return self.qty * 100 * self.current_price

    @property
    def pnl(self) -> float:
        return self.current_value - self.cost_basis

@dataclass
class TradeLog:
    """Represents a Rebalancing event (Buy/Sell underlying)."""
    date: str
    action: str  # 'Buy' or 'Sell'
    price: float
    amount: float  # $ Value
    shares: float
    fix_capital_before: float
    fix_capital_after: float
    note: str = ""

@dataclass
class TickerData:
    """Represents a single Stock/Ticker in the system."""
    symbol: str
    current_price: float
    fix_capital: float
    shares_held: float = 0.0
    cash_locked: float = 0.0 # Virtual cash pool for this ticker
    
    # Flywheel 0-7 Components
    logs: List[TradeLog] = field(default_factory=list)
    leaps: List[OptionPosition] = field(default_factory=list)
    puts: List[OptionPosition] = field(default_factory=list)
    income_shorts: List[OptionPosition] = field(default_factory=list)

    @property
    def net_liquidity(self) -> float:
        """Total Value = Shares + Cash + Options Value"""
        opt_val = sum(o.current_value for o in self.leaps + self.puts + self.income_shorts)
        return (self.shares_held * self.current_price) + self.cash_locked + opt_val

@dataclass
class SystemConfig:
    """Global Settings (Flywheel 5, 7)."""
    vix_level: float = 20.0
    risk_free_rate: float = 0.045
    collateral_yield: float = 0.05
