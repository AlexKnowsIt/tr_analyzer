from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError
from scipy.optimize import minimize

ASSET_TYPE_MAP = {
    "CRYPTOCURRENCY": "Krypto",
    "ETF": "ETF",
    "EQUITY": "Aktie",
    "BOND": "Anleihe",
    "MUTUALFUND": "Anleihe",
}

VALID_CATEGORIES = ["Krypto", "ETF", "Aktie", "Anleihe", "Sonstiges"]


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


def classify_asset(quote_type: str) -> str:
    return ASSET_TYPE_MAP.get(quote_type.upper() if quote_type else "", "Sonstiges")


def build_sector_constraints(
    category_map: Dict[str, str],
    group_bounds: Dict[str, Tuple[float, float]],
) -> Tuple[Dict, Dict, Dict]:
    """Build sector_mapper, sector_lower, sector_upper for PyPortfolioOpt."""
    sector_mapper = {}
    sector_lower = {}
    sector_upper = {}

    for category, (lo, hi) in group_bounds.items():
        if lo > 1.0:
            lo = lo / 100.0
        if hi > 1.0:
            hi = hi / 100.0
        if lo > hi:
            raise ValueError(f"Group bound min {lo} > max {hi} for category '{category}'")
        sector_lower[category] = lo
        sector_upper[category] = hi

    for isin, category in category_map.items():
        sector_mapper[isin] = category

    return sector_mapper, sector_lower, sector_upper


def run_optimization(
    price_data: pd.DataFrame,
    objective: str,
    target_value: float,
    individual_constraints: List[Tuple[int, str, float]],
    sector_mapper: Dict,
    sector_lower: Dict,
    sector_upper: Dict,
) -> Dict:
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    ef = EfficientFrontier(mu, S)

    tickers = list(price_data.columns)
    for idx, op, val in individual_constraints:
        if op == "<=":
            ef.add_constraint(lambda w, i=idx, v=val: w[i] <= v)
        else:
            ef.add_constraint(lambda w, i=idx, v=val: w[i] >= v)

    if sector_mapper:
        ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

    if objective == "risk_parity":
        return compute_risk_parity(price_data)

    try:
        if objective == "max_sharpe":
            ef.max_sharpe()
        elif objective == "min_volatility":
            ef.min_volatility()
        elif objective == "efficient_return":
            ef.efficient_return(target_return=float(target_value))
        elif objective == "efficient_risk":
            ef.efficient_risk(target_volatility=float(target_value))
        else:
            raise ValueError(f"Unknown objective: {objective}")
    except OptimizationError as e:
        raise ValueError(str(e))

    weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance()
    return {
        "weights": {k: float(v) for k, v in weights.items()},
        "expected_return": float(ret),
        "volatility": float(vol),
        "sharpe": float(sharpe),
    }


def compute_performance(price_data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    returns = price_data.pct_change().dropna()

    weights_series = pd.Series(weights).reindex(price_data.columns).fillna(0.0)
    weights_series = weights_series / weights_series.sum()

    port_returns = returns.dot(weights_series)

    cum = (1 + port_returns).cumprod()
    rolling_max = cum.expanding().max()
    drawdown = (cum - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    n_years = len(port_returns) / 252
    total_return = float(cum.iloc[-1] - 1)
    cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0
    volatility = float(port_returns.std() * np.sqrt(252))
    sharpe = float((port_returns.mean() / port_returns.std()) * np.sqrt(252)) if port_returns.std() > 0 else 0.0

    window = 252
    rolling_sharpe_series = port_returns.rolling(window).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else np.nan
    )

    dates = [d.strftime("%Y-%m-%d") for d in port_returns.index]
    cumulative_return = [float(v) for v in (cum - 1)]
    rolling_sharpe = [None if np.isnan(v) else float(v) for v in rolling_sharpe_series]

    return {
        "dates": dates,
        "cumulative_return": cumulative_return,
        "rolling_sharpe": rolling_sharpe,
        "cagr": cagr,
        "volatility": volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def compute_frontier(price_data: pd.DataFrame, n: int = 500) -> Dict:
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    S_arr = S.values
    n_assets = len(mu)
    mu_arr = mu.values

    rng = np.random.default_rng(42)
    rets, vols, sharpes = [], [], []
    for _ in range(n):
        w = rng.dirichlet(np.ones(n_assets))
        r = float(w @ mu_arr)
        v = float(np.sqrt(w @ S_arr @ w))
        rets.append(r)
        vols.append(v)
        sharpes.append(r / v if v > 0 else 0.0)

    frontier_rets, frontier_vols = [], []
    try:
        min_ret = float(mu_arr.min()) * 1.01
        max_ret = float(mu_arr.max()) * 0.99
        for target in np.linspace(min_ret, max_ret, 30):
            try:
                ef = EfficientFrontier(mu, S)
                ef.efficient_return(target)
                w = np.array(list(ef.clean_weights().values()))
                frontier_vols.append(float(np.sqrt(w @ S_arr @ w)))
                frontier_rets.append(float(target))
            except (OptimizationError, Exception):
                pass
    except Exception:
        pass

    return {
        "returns": rets,
        "vols": vols,
        "sharpes": sharpes,
        "frontier_returns": frontier_rets,
        "frontier_vols": frontier_vols,
    }


def simulate_savings_plan(
    monthly_amount: float,
    years: int,
    annual_return: float,
    annual_volatility: float,
    annual_savings_increase: float = 0.0,
    inflation_rate: float = 0.02,
    tax_rate: float = 0.26375,
) -> Dict:
    monthly_base = (1 + annual_return) ** (1 / 12) - 1
    monthly_bull = (1 + annual_return + annual_volatility) ** (1 / 12) - 1
    monthly_bear = (1 + max(annual_return - annual_volatility, -0.5)) ** (1 / 12) - 1

    years_list = list(range(1, years + 1))
    invested = []
    base_gross, bull_gross, bear_gross = [], [], []
    base_net, real_value = [], []
    monthly_amounts = []

    val_base = val_bull = val_bear = 0.0
    total_invested = 0.0

    for yr in range(1, years + 1):
        amt = monthly_amount * (1 + annual_savings_increase) ** (yr - 1)
        monthly_amounts.append(round(amt, 2))
        for _ in range(12):
            val_base = val_base * (1 + monthly_base) + amt
            val_bull = val_bull * (1 + monthly_bull) + amt
            val_bear = val_bear * (1 + monthly_bear) + amt
            total_invested += amt

        invested.append(round(total_invested, 2))
        base_gross.append(round(val_base, 2))
        bull_gross.append(round(val_bull, 2))
        bear_gross.append(round(val_bear, 2))

        gains = max(val_base - total_invested, 0.0)
        net = val_base - gains * tax_rate
        base_net.append(round(net, 2))

        deflator = (1 + inflation_rate) ** yr
        real_value.append(round(net / deflator, 2))

    return {
        "years_list": years_list,
        "invested": invested,
        "base_gross": base_gross,
        "bull_gross": bull_gross,
        "bear_gross": bear_gross,
        "base_net": base_net,
        "real_value": real_value,
        "monthly_amounts": monthly_amounts,
    }


def compute_risk_contributions(price_data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    S = risk_models.sample_cov(price_data)
    w = pd.Series(weights).reindex(price_data.columns).fillna(0.0).values
    w = w / w.sum()
    S_arr = S.values
    port_var = float(w @ S_arr @ w)
    port_vol = float(np.sqrt(port_var))
    marginal = (S_arr @ w) / port_vol
    component = w * marginal
    pct = component / port_vol
    return {
        "tickers": list(price_data.columns),
        "marginal": marginal.tolist(),
        "component": component.tolist(),
        "percentage": pct.tolist(),
        "portfolio_vol": port_vol,
    }


def compute_var_cvar(
    price_data: pd.DataFrame,
    weights: Dict[str, float],
    confidence_levels: tuple = (0.95, 0.99),
) -> Dict:
    returns = price_data.pct_change().dropna()
    w = pd.Series(weights).reindex(price_data.columns).fillna(0.0)
    w = w / w.sum()
    port_returns = returns.dot(w)
    result: Dict = {
        "dates": [d.strftime("%Y-%m-%d") for d in port_returns.index],
        "daily_returns": [float(v) for v in port_returns],
    }
    for cl in confidence_levels:
        key = str(int(cl * 100))
        var = float(np.percentile(port_returns, (1 - cl) * 100))
        tail = port_returns[port_returns <= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        result[f"var_{key}"] = var
        result[f"cvar_{key}"] = cvar
    return result


def compute_beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict:
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    x = benchmark_returns.loc[common].values
    y = portfolio_returns.loc[common].values
    if len(x) < 2 or np.var(x) == 0:
        return {"beta": 1.0, "alpha": 0.0, "r_squared": 0.0}
    beta = float(np.cov(x, y)[0, 1] / np.var(x))
    alpha_daily = float(np.mean(y) - beta * np.mean(x))
    alpha_annual = float((1 + alpha_daily) ** 252 - 1)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    return {"beta": beta, "alpha": alpha_annual, "r_squared": r2}


def compute_return_attribution(price_data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    w = pd.Series(weights).reindex(price_data.columns).fillna(0.0)
    w = w / w.sum()
    asset_returns = price_data.iloc[-1] / price_data.iloc[0] - 1
    contributions = w * asset_returns
    return {
        "tickers": list(price_data.columns),
        "asset_returns": [float(v) for v in asset_returns],
        "contributions": [float(v) for v in contributions],
        "total": float(contributions.sum()),
    }


_STRESS_SCENARIOS = {
    "2022 Zinsanstieg": ("2022-01-03", "2022-10-13"),
    "COVID-Crash 2020": ("2020-02-19", "2020-03-23"),
    "Finanzkrise 2008": ("2008-09-01", "2009-03-31"),
}


def run_stress_test(price_data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    w = pd.Series(weights).reindex(price_data.columns).fillna(0.0)
    w = w / w.sum()
    results = {}
    for name, (start, end) in _STRESS_SCENARIOS.items():
        slice_ = price_data.loc[start:end]
        if len(slice_) < 2:
            results[name] = {"available": False, "start": start, "end": end}
            continue
        asset_rets = slice_.pct_change().dropna()
        port_rets = asset_rets.dot(w)
        cum = (1 + port_rets).cumprod()
        total_return = float(cum.iloc[-1] - 1)
        rolling_max = cum.expanding().max()
        max_dd = float(((cum - rolling_max) / rolling_max).min())
        results[name] = {
            "available": True,
            "start": start,
            "end": end,
            "total_return": total_return,
            "max_drawdown": max_dd,
        }
    return results


def simulate_mc_paths(
    price_data: pd.DataFrame,
    weights: Dict[str, float],
    years: int,
    n_paths: int,
    start_value: float,
) -> Dict:
    mu = expected_returns.mean_historical_return(price_data).values
    S = risk_models.sample_cov(price_data).values
    w = pd.Series(weights).reindex(price_data.columns).fillna(0.0).values
    w = w / w.sum()
    n_assets = len(mu)
    n_steps = years * 252
    dt = 1.0 / 252

    try:
        L = np.linalg.cholesky(S + np.eye(n_assets) * 1e-8)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(S)))

    rng = np.random.default_rng(0)
    # Vectorized: shape (n_paths, n_steps, n_assets)
    Z = rng.standard_normal((n_paths, n_steps, n_assets))
    # Correlated noise: (L @ Z[p,t,:]) for each path/step
    dW = (L @ Z.reshape(-1, n_assets).T).T.reshape(n_paths, n_steps, n_assets)
    drift = (mu - 0.5 * np.diag(S)) * dt  # (n_assets,)
    daily_asset = drift + dW * np.sqrt(dt)  # (n_paths, n_steps, n_assets)
    port_daily = daily_asset @ w  # (n_paths, n_steps)
    # Log-return path → price path
    paths = start_value * np.exp(np.cumsum(np.log1p(port_daily), axis=1))  # (n_paths, n_steps)

    pcts = {}
    for p in [5, 25, 50, 75, 95]:
        pcts[str(p)] = np.percentile(paths, p, axis=0).tolist()

    day_labels = [f"Y{(i+1)/252:.2f}" for i in range(n_steps)]
    return {
        "percentiles": pcts,
        "n_steps": n_steps,
        "years": years,
        "start_value": start_value,
    }


def compute_risk_parity(price_data: pd.DataFrame) -> Dict:
    S = risk_models.sample_cov(price_data).values
    mu = expected_returns.mean_historical_return(price_data)
    mu_arr = mu.values
    n = S.shape[0]
    tickers = list(price_data.columns)

    def erc_objective(w):
        var = float(w @ S @ w)
        if var <= 0:
            return 1e10
        rc = w * (S @ w) / var
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    w0 = np.ones(n) / n
    res = minimize(
        erc_objective,
        w0,
        bounds=[(1e-6, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = res.x / res.x.sum()
    ret = float(w @ mu_arr)
    vol = float(np.sqrt(w @ S @ w))
    return {
        "weights": {t: float(wi) for t, wi in zip(tickers, w)},
        "expected_return": ret,
        "volatility": vol,
        "sharpe": ret / vol if vol > 0 else 0.0,
    }
