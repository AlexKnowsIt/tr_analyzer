from typing import List, Tuple

def parse_constraints(constraint_str: str, tickers: List[str]) -> List[Tuple[int, str, float]]:
    """Parse constraint string into (index, op, value) tuples."""
    constraints = []
    for part in constraint_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "<=" in part:
            ticker, val = part.split("<=")
            op = "<="
        elif ">=" in part:
            ticker, val = part.split(">=")
            op = ">="
        else:
            continue
        ticker = ticker.strip()
        val = float(val.strip())
        idx = tickers.index(ticker)
        constraints.append((idx, op, val))
    return constraints
