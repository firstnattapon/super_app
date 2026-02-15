import json
import re
import pandas as pd
from typing import List, Dict
from .models import TickerData, TradeLog

class DataManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tickers: Dict[str, TickerData] = {}
        self.raw_data = []

    def load_data(self):
        """Loads JSON data and parses it into TickerData objects."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            for item in self.raw_data:
                symbol = item.get("ticker", "UNKNOWN")
                
                # Parse 'Final' string for current state: "Price, Capital, PnL"
                # Example: "6.88, 1500, 0"
                final_str = item.get("Final", "0,0,0")
                parts = [float(x.strip()) if x.strip() else 0 for x in final_str.split(',')]
                current_price = parts[0] if len(parts) > 0 else 0
                fix_capital = parts[1] if len(parts) > 1 else 0
                
                # Initialize Ticker Object
                ticker_obj = TickerData(
                    symbol=symbol,
                    current_price=current_price,
                    fix_capital=fix_capital
                )
                
                # Parse History Strings
                self._parse_history_logs(item, ticker_obj)
                
                self.tickers[symbol] = ticker_obj
                
        except FileNotFoundError:
            print("File not found.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def _parse_history_logs(self, json_item: dict, ticker: TickerData):
        """Parses 'history_X' and 'history_X.1' keys."""
        # Generic Regex to find strings like "history_1", "history_2"
        # And their details "history_1.1"
        
        # We iterate through keys to find 'history_' patterns
        keys = sorted(json_item.keys())
        history_keys = [k for k in keys if k.startswith("history_") and "." not in k]
        
        for key in history_keys:
            # Step description: "ราคาอ้างอิง: 1.26 → 25.2, ทุนคงที่: 1500 → 1500..."
            desc = json_item.get(key, "")
            
            # Detail equation: "-4493.598 += (1500 * ln(25.2 / 1.26))..."
            detail_key = key + ".1"
            equation = json_item.get(detail_key, "")
            
            # Simple parsing logic (can be enhanced)
            if "→" in desc:
                # Attempt to extract Rebalance Action
                # This is a bit abstract in the text, assume it's a rebalance event
                log = TradeLog(
                    date="Unknown", # Original data has no date
                    action="Rebalance",
                    price=0.0, # Placeholder
                    amount=0.0,
                    shares=0.0,
                    fix_capital_before=0.0,
                    fix_capital_after=0.0,
                    note=f"{desc} | {equation}"
                )
                ticker.logs.append(log)

    def get_all_tickers(self) -> List[TickerData]:
        return list(self.tickers.values())

    def get_pool_cf(self) -> float:
        """Calculates Total Cash Flow from all tickers (sum of Net PnL)."""
        # In original JSON, 'beta_momory' -> "Net: +1027.28"
        total_cf = 0.0
        for item in self.raw_data:
            beta_mem = item.get("beta_momory", "")
            match = re.search(r"Net:\s*([+\-]?[\d,.]+)", beta_mem)
            if match:
                val_str = match.group(1).replace(',', '')
                total_cf += float(val_str)
        return total_cf
