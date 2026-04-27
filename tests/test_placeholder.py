import numpy as np
import pandas as pd
import pytest

from optimizer import (
    build_sector_constraints,
    classify_asset,
    compute_frontier,
    compute_performance,
    parse_constraints,
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
