import numpy as np
import pandas as pd
import pytest

from optimizer import (
    build_sector_constraints,
    classify_asset,
    compute_frontier,
    compute_performance,
    parse_constraints,
    run_optimization,
    simulate_savings_plan,
)


def test_parse_constraints():
    tickers = ["AAA", "BBB"]
    constraints = parse_constraints("AAA<=0.2,BBB>=0.1", tickers)
    assert constraints == [(0, "<=", 0.2), (1, ">=", 0.1)]


def test_classify_asset():
    assert classify_asset("CRYPTOCURRENCY") == "Krypto"
    assert classify_asset("ETF") == "ETF"
    assert classify_asset("EQUITY") == "Aktie"
    assert classify_asset("BOND") == "Anleihe"
    assert classify_asset("unknown_type") == "Sonstiges"
    assert classify_asset("") == "Sonstiges"


def test_build_sector_constraints_basic():
    category_map = {"IE001": "ETF", "IE002": "Krypto"}
    group_bounds = {"ETF": (0.2, 0.8), "Krypto": (0.0, 0.1)}
    mapper, lower, upper = build_sector_constraints(category_map, group_bounds)
    assert mapper == {"IE001": "ETF", "IE002": "Krypto"}
    assert lower["ETF"] == pytest.approx(0.2)
    assert upper["Krypto"] == pytest.approx(0.1)


def test_build_sector_constraints_percent_conversion():
    category_map = {"IE001": "ETF"}
    group_bounds = {"ETF": (30.0, 80.0)}
    _, lower, upper = build_sector_constraints(category_map, group_bounds)
    assert lower["ETF"] == pytest.approx(0.30)
    assert upper["ETF"] == pytest.approx(0.80)


def test_build_sector_constraints_invalid_bounds():
    with pytest.raises(ValueError, match="min.*>.*max"):
        build_sector_constraints({"IE001": "ETF"}, {"ETF": (0.9, 0.1)})


def _make_price_data(n_days=500, n_assets=3, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    tickers = [f"T{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_compute_performance_shape():
    price_data = _make_price_data()
    weights = {"T0": 0.5, "T1": 0.3, "T2": 0.2}
    result = compute_performance(price_data, weights)

    expected_keys = {"dates", "cumulative_return", "rolling_sharpe", "cagr", "volatility", "max_drawdown", "sharpe"}
    assert expected_keys == set(result.keys())

    n = len(price_data) - 1
    assert len(result["dates"]) == n
    assert len(result["cumulative_return"]) == n
    assert len(result["rolling_sharpe"]) == n


def test_compute_performance_max_drawdown_nonpositive():
    price_data = _make_price_data()
    weights = {"T0": 0.5, "T1": 0.3, "T2": 0.2}
    result = compute_performance(price_data, weights)
    assert result["max_drawdown"] <= 0


def test_compute_frontier_shape():
    price_data = _make_price_data(n_days=600, n_assets=4)
    result = compute_frontier(price_data, n=50)
    assert set(result.keys()) == {"returns", "vols", "sharpes", "frontier_returns", "frontier_vols"}
    assert len(result["returns"]) == 50
    assert len(result["vols"]) == 50
    assert len(result["sharpes"]) == 50
    assert len(result["frontier_returns"]) == len(result["frontier_vols"])


def test_compute_frontier_sharpe_range():
    price_data = _make_price_data(n_days=600, n_assets=4)
    result = compute_frontier(price_data, n=100)
    assert all(s >= -10 for s in result["sharpes"])


def test_simulate_savings_plan_base():
    result = simulate_savings_plan(
        monthly_amount=500, years=10, annual_return=0.07,
        annual_volatility=0.15, annual_savings_increase=0.0,
        inflation_rate=0.02, tax_rate=0.26375,
    )
    assert result["invested"][-1] == pytest.approx(500 * 12 * 10, rel=0.01)
    assert result["base_gross"][-1] > result["invested"][-1]
    assert len(result["years_list"]) == 10
    assert result["invested"] == sorted(result["invested"])


def test_simulate_savings_plan_bear_lt_base():
    result = simulate_savings_plan(
        monthly_amount=300, years=5, annual_return=0.07,
        annual_volatility=0.12,
    )
    for base, bear in zip(result["base_gross"], result["bear_gross"]):
        assert bear <= base


def test_simulate_savings_plan_increase():
    result = simulate_savings_plan(
        monthly_amount=200, years=5, annual_return=0.07,
        annual_volatility=0.10, annual_savings_increase=0.05,
    )
    assert result["monthly_amounts"][-1] > result["monthly_amounts"][0]


def test_simulate_savings_plan_tax_reduces_net():
    result = simulate_savings_plan(
        monthly_amount=500, years=10, annual_return=0.07,
        annual_volatility=0.15, tax_rate=0.26375,
    )
    for gross, net in zip(result["base_gross"], result["base_net"]):
        assert net <= gross


def test_simulate_savings_plan_bull_gt_base():
    result = simulate_savings_plan(
        monthly_amount=500, years=10, annual_return=0.07,
        annual_volatility=0.15,
    )
    for bull, base in zip(result["bull_gross"], result["base_gross"]):
        assert bull >= base


def test_simulate_savings_plan_real_le_net():
    result = simulate_savings_plan(
        monthly_amount=500, years=20, annual_return=0.07,
        annual_volatility=0.15, inflation_rate=0.02,
    )
    for real, net in zip(result["real_value"], result["base_net"]):
        assert real <= net


def test_simulate_savings_plan_zero_return():
    result = simulate_savings_plan(
        monthly_amount=100, years=5, annual_return=0.0,
        annual_volatility=0.0, inflation_rate=0.0, tax_rate=0.0,
    )
    assert result["base_gross"][-1] == pytest.approx(100 * 12 * 5, rel=0.001)


# ---- parse_constraints edge cases ----

def test_parse_constraints_empty():
    assert parse_constraints("", ["AAA"]) == []
    assert parse_constraints("  ,  , ", ["AAA"]) == []


def test_parse_constraints_whitespace():
    tickers = ["AAA", "BBB"]
    result = parse_constraints(" AAA <= 0.3 , BBB >= 0.05 ", tickers)
    assert result == [(0, "<=", 0.3), (1, ">=", 0.05)]


def test_parse_constraints_unknown_ticker():
    with pytest.raises(ValueError):
        parse_constraints("ZZZ<=0.5", ["AAA"])


# ---- compute_performance edge cases ----

def test_compute_performance_rolling_sharpe_prefix_none():
    price_data = _make_price_data(n_days=500)
    result = compute_performance(price_data, {"T0": 0.5, "T1": 0.3, "T2": 0.2})
    # first 251 entries must be None (window=252, need 252 obs → 251 NaN)
    assert all(v is None for v in result["rolling_sharpe"][:251])
    assert any(v is not None for v in result["rolling_sharpe"][251:])


def test_compute_performance_unnormalized_weights():
    price_data = _make_price_data()
    # weights sum to 2.0 — should be normalized internally
    r1 = compute_performance(price_data, {"T0": 1.0, "T1": 0.6, "T2": 0.4})
    r2 = compute_performance(price_data, {"T0": 0.5, "T1": 0.3, "T2": 0.2})
    assert r1["cagr"] == pytest.approx(r2["cagr"], rel=1e-6)


# ---- compute_frontier edge cases ----

def test_compute_frontier_vols_positive():
    price_data = _make_price_data(n_days=600, n_assets=4)
    result = compute_frontier(price_data, n=50)
    assert all(v > 0 for v in result["vols"])


def test_compute_frontier_no_nan():
    price_data = _make_price_data(n_days=600, n_assets=4)
    result = compute_frontier(price_data, n=50)
    assert all(np.isfinite(v) for v in result["returns"])
    assert all(np.isfinite(v) for v in result["vols"])
    assert all(np.isfinite(v) for v in result["sharpes"])


# ---- run_optimization ----

def _make_price_data_opt(n_days=800, n_assets=5, seed=7):
    rng = np.random.default_rng(seed)
    # positive drift so max_sharpe is well-defined
    returns = rng.normal(0.0006, 0.01, size=(n_days, n_assets))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    tickers = [f"T{i}" for i in range(n_assets)]
    dates = pd.date_range("2019-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=tickers)


def test_run_optimization_max_sharpe_weights_sum():
    price_data = _make_price_data_opt()
    result = run_optimization(price_data, "max_sharpe", None, [], {}, {}, {})
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-4)
    assert result["sharpe"] > 0
    assert result["volatility"] > 0
    assert result["expected_return"] > 0


def test_run_optimization_min_vol_lt_max_sharpe_vol():
    price_data = _make_price_data_opt()
    r_sharpe = run_optimization(price_data, "max_sharpe", None, [], {}, {}, {})
    r_vol = run_optimization(price_data, "min_volatility", None, [], {}, {}, {})
    assert r_vol["volatility"] <= r_sharpe["volatility"]


def test_run_optimization_individual_constraint_respected():
    price_data = _make_price_data_opt()
    tickers = list(price_data.columns)
    # cap T0 at 20%
    constraints = [(0, "<=", 0.20)]
    result = run_optimization(price_data, "max_sharpe", None, constraints, {}, {}, {})
    assert result["weights"][tickers[0]] <= 0.20 + 1e-4


def test_run_optimization_invalid_objective():
    price_data = _make_price_data_opt()
    with pytest.raises(ValueError):
        run_optimization(price_data, "nonexistent_objective", None, [], {}, {}, {})
