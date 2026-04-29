"""Flask REST API for portfolio analysis."""

import asyncio
import io
import json
import secrets
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from optimizer import (
    VALID_CATEGORIES,
    build_sector_constraints,
    classify_asset,
    compute_beta_alpha,
    compute_buy_recommendation,
    compute_etf_overlap,
    compute_frontier,
    compute_performance,
    compute_return_attribution,
    compute_risk_contributions,
    compute_risk_parity,
    compute_var_cvar,
    parse_constraints,
    run_optimization,
    run_stress_test,
    simulate_mc_paths,
    simulate_savings_plan,
)
from pytr.api import TradeRepublicApi
from pytr.portfolio import Portfolio

app = Flask(__name__)
PORTFOLIO_FILE = Path(__file__).parent / "portfolio.csv"

_price_cache: dict = {}
_CACHE_TTL = 3600

_tr_sessions: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_portfolio() -> pd.DataFrame:
    if not PORTFOLIO_FILE.exists():
        return None
    df = pd.read_csv(PORTFOLIO_FILE, sep=";")
    required = {"Name", "ISIN", "netValue"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    return df


def _fetch_prices(tickers: list) -> pd.DataFrame:
    key = frozenset(tickers)
    cached = _price_cache.get(key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]

    raw = yf.download(tickers=tickers, period="5y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.dropna(how="all")
    _price_cache[key] = (time.time(), prices)
    return prices


def _fetch_info(tickers: list) -> list:
    rows = []
    for isin in tickers:
        try:
            info = yf.Ticker(isin).info
        except Exception:
            info = {}
        rows.append({
            "isin": isin,
            "name": info.get("longName") or info.get("shortName", isin),
            "region": info.get("country", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "category": classify_asset(info.get("quoteType", "")),
        })
    return rows


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/portfolio")
def api_portfolio():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    tickers = df["ISIN"].tolist()
    info_list = _fetch_info(tickers)
    info_map = {r["isin"]: r for r in info_list}

    total = float(df["netValue"].sum())
    holdings = []
    for _, row in df.iterrows():
        isin = row["ISIN"]
        info = info_map.get(isin, {})
        net = float(row["netValue"])
        holdings.append({
            "isin": isin,
            "name": row.get("Name", isin),
            "region": info.get("region", "Unknown"),
            "sector": info.get("sector", "Unknown"),
            "category": info.get("category", "Sonstiges"),
            "net_value": net,
            "weight": round(net / total, 6) if total > 0 else 0,
        })

    return jsonify({"holdings": holdings, "total": total})


@app.route("/api/prices")
def api_prices():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    return jsonify({"prices": prices.to_dict(orient="list"), "dates": [d.strftime("%Y-%m-%d") for d in prices.index]})


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    body = request.get_json(force=True) or {}
    objective = body.get("objective", "max_sharpe")
    target_value = body.get("target_value")
    constraint_str = body.get("constraints", "")
    category_overrides = body.get("category_overrides", {})
    group_bounds_raw = body.get("group_bounds", {})

    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422

    try:
        individual = parse_constraints(constraint_str, list(prices.columns))
    except (ValueError, IndexError) as e:
        return jsonify({"error": f"Invalid constraints: {e}"}), 422

    info_list = _fetch_info(tickers)
    info_map = {r["isin"]: r for r in info_list}
    category_map = {}
    for isin in tickers:
        category_map[isin] = category_overrides.get(isin) or info_map.get(isin, {}).get("category", "Sonstiges")

    group_bounds = {}
    for cat, bounds in group_bounds_raw.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            group_bounds[cat] = (float(bounds[0]), float(bounds[1]))

    try:
        sector_mapper, sector_lower, sector_upper = build_sector_constraints(category_map, group_bounds)
        result = run_optimization(prices, objective, target_value, individual, sector_mapper, sector_lower, sector_upper)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    return jsonify(result)


@app.route("/api/performance")
def api_performance():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    weights_param = request.args.get("weights")
    tickers = df["ISIN"].tolist()

    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422

    if weights_param:
        try:
            weights = json.loads(weights_param)
        except json.JSONDecodeError:
            return jsonify({"error": "invalid weights JSON"}), 422
    else:
        total = float(df["netValue"].sum())
        weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}

    try:
        result = compute_performance(prices, weights)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    return jsonify(result)


@app.route("/api/charts/allocation")
def api_charts_allocation():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    tickers = df["ISIN"].tolist()
    info_list = _fetch_info(tickers)
    info_map = {r["isin"]: r for r in info_list}

    total = float(df["netValue"].sum())
    rows = []
    for _, row in df.iterrows():
        isin = row["ISIN"]
        info = info_map.get(isin, {})
        rows.append({
            "name": row.get("Name", isin),
            "isin": isin,
            "category": info.get("category", "Sonstiges"),
            "sector": info.get("sector", "Unknown"),
            "region": info.get("region", "Unknown"),
            "net_value": float(row["netValue"]),
        })
    merged = pd.DataFrame(rows)

    try:
        prices = _fetch_prices(tickers)
        prices = prices.reindex(columns=tickers).dropna()
        corr = prices.pct_change().dropna().corr()
        corr_data = {
            "z": corr.values.tolist(),
            "x": list(corr.columns),
            "y": list(corr.index),
        }
    except Exception:
        corr_data = None

    sector_data = merged.groupby("sector")["net_value"].sum().reset_index()
    region_data = merged.groupby("region")["net_value"].sum().reset_index()
    category_data = merged.groupby("category")["net_value"].sum().reset_index()

    return jsonify({
        "sector": sector_data.to_dict(orient="list"),
        "region": region_data.to_dict(orient="list"),
        "category": category_data.to_dict(orient="list"),
        "correlation": corr_data,
        "holdings": rows,
    })


@app.route("/api/tr/login", methods=["POST"])
def api_tr_login():
    body = request.get_json(force=True) or {}
    phone = body.get("phone", "").strip()
    pin = body.get("pin", "").strip()
    if not phone or not pin:
        return jsonify({"error": "phone and pin required"}), 422

    try:
        tr = TradeRepublicApi(phone_no=phone, pin=pin)
        countdown = tr.initiate_weblogin()
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    session_id = secrets.token_urlsafe(16)
    _tr_sessions[session_id] = tr
    return jsonify({"session_id": session_id, "countdown": int(countdown)})


@app.route("/api/tr/verify", methods=["POST"])
def api_tr_verify():
    body = request.get_json(force=True) or {}
    session_id = body.get("session_id", "")
    code = str(body.get("code", "")).strip()

    tr = _tr_sessions.pop(session_id, None)
    if tr is None:
        return jsonify({"error": "session not found or expired"}), 404
    if not code:
        return jsonify({"error": "code required"}), 422

    try:
        tr.complete_weblogin(code)
        pf = Portfolio(tr, output=str(PORTFOLIO_FILE))
        asyncio.run(pf.portfolio_loop())
        pf.portfolio_to_csv()
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    _price_cache.clear()
    return jsonify({"status": "ok", "count": len(pf.portfolio)})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 422
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 422

    content = f.read().decode("utf-8")
    try:
        df = pd.read_csv(io.StringIO(content), sep=";")
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 422

    required = {"Name", "ISIN", "netValue"}
    if not required.issubset(df.columns):
        return jsonify({"error": f"Missing columns: {required - set(df.columns)}"}), 422

    PORTFOLIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PORTFOLIO_FILE, sep=";", index=False)
    _price_cache.clear()
    return jsonify({"status": "ok", "count": len(df)})


BENCHMARK_TICKERS = {
    "MSCI World": "IWDA.AS",
    "DAX": "^GDAXI",
    "S&P 500": "^GSPC",
}


@app.route("/api/benchmark")
def api_benchmark():
    ticker = request.args.get("ticker", "IWDA.AS")
    period = request.args.get("period", "5y")
    try:
        prices = _fetch_prices([ticker])
        col = prices.columns[0]
        series = prices[col].dropna()
        returns = series.pct_change().dropna()
        cum = (1 + returns).cumprod() - 1
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    name = next((k for k, v in BENCHMARK_TICKERS.items() if v == ticker), ticker)
    return jsonify({
        "dates": [d.strftime("%Y-%m-%d") for d in returns.index],
        "cumulative_return": [float(v) for v in cum],
        "name": name,
    })


@app.route("/api/frontier", methods=["POST"])
def api_frontier():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    body = request.get_json(force=True) or {}
    current_weights = body.get("current_weights")
    opt_weights = body.get("opt_weights")

    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422

    try:
        from pypfopt import expected_returns, risk_models
        result = compute_frontier(prices)
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        S_arr = S.values
        mu_arr = mu.values
        import numpy as np
        for label, weights_dict in [("current", current_weights), ("optimized", opt_weights)]:
            if weights_dict:
                w = pd.Series(weights_dict).reindex(prices.columns).fillna(0.0).values
                w = w / w.sum() if w.sum() > 0 else w
                result[label] = {
                    "vol": float(np.sqrt(w @ S_arr @ w)),
                    "ret": float(w @ mu_arr),
                }
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    return jsonify(result)


@app.route("/api/simulation", methods=["POST"])
def api_simulation():
    body = request.get_json(force=True) or {}

    monthly_amount = float(body.get("monthly_amount", 500))
    years = int(body.get("years", 20))
    annual_savings_increase = float(body.get("annual_savings_increase", 0.0))
    inflation_rate = float(body.get("inflation_rate", 0.02))
    tax_rate = float(body.get("tax_rate", 0.26375))

    annual_return = body.get("annual_return")
    annual_volatility = body.get("annual_volatility")

    if annual_return is None or annual_volatility is None:
        df = _load_portfolio()
        if df is not None:
            try:
                tickers = df["ISIN"].tolist()
                prices = _fetch_prices(tickers)
                prices = prices.reindex(columns=tickers).dropna()
                total = float(df["netValue"].sum())
                weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}
                perf = compute_performance(prices, weights)
                if annual_return is None:
                    annual_return = perf["cagr"]
                if annual_volatility is None:
                    annual_volatility = perf["volatility"]
            except Exception:
                pass

    annual_return = float(annual_return) if annual_return is not None else 0.07
    annual_volatility = float(annual_volatility) if annual_volatility is not None else 0.15

    try:
        result = simulate_savings_plan(
            monthly_amount=monthly_amount,
            years=years,
            annual_return=annual_return,
            annual_volatility=annual_volatility,
            annual_savings_increase=annual_savings_increase,
            inflation_rate=inflation_rate,
            tax_rate=tax_rate,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    result["annual_return_used"] = annual_return
    result["annual_volatility_used"] = annual_volatility
    return jsonify(result)


@app.route("/api/risk")
def api_risk():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404

    weights_param = request.args.get("weights")
    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422

    if weights_param:
        try:
            weights = json.loads(weights_param)
        except json.JSONDecodeError:
            return jsonify({"error": "invalid weights JSON"}), 422
    else:
        total = float(df["netValue"].sum())
        weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}

    try:
        risk_contrib = compute_risk_contributions(prices, weights)
        var_data = compute_var_cvar(prices, weights)
        # beta vs MSCI World
        bench_prices = _fetch_prices(["IWDA.AS"])
        bench_col = bench_prices.columns[0]
        bench_ret = bench_prices[bench_col].pct_change().dropna()
        port_returns = prices.pct_change().dropna().dot(
            pd.Series(weights).reindex(prices.columns).fillna(0)
        )
        port_returns_norm = port_returns / port_returns.abs().sum() * port_returns.abs().sum()
        beta_data = compute_beta_alpha(port_returns, bench_ret)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    return jsonify({**risk_contrib, **var_data, **beta_data})


@app.route("/api/attribution")
def api_attribution():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404
    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422
    total = float(df["netValue"].sum())
    weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}
    try:
        result = compute_return_attribution(prices, weights)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    # enrich with names
    name_map = {row["ISIN"]: row.get("Name", row["ISIN"]) for _, row in df.iterrows()}
    result["names"] = [name_map.get(t, t) for t in result["tickers"]]
    return jsonify(result)


@app.route("/api/stress")
def api_stress():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404
    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422
    total = float(df["netValue"].sum())
    weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}
    try:
        result = run_stress_test(prices, weights)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    return jsonify(result)


@app.route("/api/monte-carlo", methods=["POST"])
def api_monte_carlo():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404
    body = request.get_json(force=True) or {}
    years = int(body.get("years", 10))
    n_paths = int(body.get("n_paths", 200))
    start_value = float(body.get("start_value", 10000))
    weights_body = body.get("weights")
    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    prices = prices.reindex(columns=tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422
    if weights_body:
        weights = weights_body
    else:
        total = float(df["netValue"].sum())
        weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}
    try:
        result = simulate_mc_paths(prices, weights, years=years, n_paths=n_paths, start_value=start_value)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    return jsonify(result)


@app.route("/api/income")
def api_income():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404
    rows = []
    total_dividend = 0.0
    total_ter_cost = 0.0
    for _, row in df.iterrows():
        isin = row["ISIN"]
        net = float(row["netValue"])
        try:
            info = yf.Ticker(isin).info
        except Exception:
            info = {}
        div_yield = float(info.get("dividendYield") or 0)
        ter = float(info.get("annualReportExpenseRatio") or 0)
        annual_div = net * div_yield
        annual_ter = net * ter
        total_dividend += annual_div
        total_ter_cost += annual_ter
        rows.append({
            "isin": isin,
            "name": row.get("Name", isin),
            "net_value": net,
            "dividend_yield": div_yield,
            "annual_dividend": round(annual_div, 2),
            "ter": ter,
            "annual_ter_cost": round(annual_ter, 2),
        })
    return jsonify({
        "holdings": rows,
        "total_annual_dividend": round(total_dividend, 2),
        "total_annual_ter_cost": round(total_ter_cost, 2),
        "portfolio_weighted_ter": round(total_ter_cost / float(df["netValue"].sum()), 6) if df["netValue"].sum() > 0 else 0,
    })


@app.route("/api/tax-estimate")
def api_tax_estimate():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404
    required = {"avgCost", "quantity"}
    if not required.issubset(df.columns):
        return jsonify({"error": f"CSV missing columns for tax estimate: {required - set(df.columns)}"}), 422
    tickers = df["ISIN"].tolist()
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    latest = prices.iloc[-1] if not prices.empty else pd.Series()
    TAX_RATE = 0.26375
    rows = []
    total_gain = 0.0
    total_tax = 0.0
    for _, row in df.iterrows():
        isin = row["ISIN"]
        avg_cost = float(row["avgCost"])
        qty = float(row["quantity"])
        current_price = float(latest.get(isin, avg_cost))
        gain = (current_price - avg_cost) * qty
        tax = max(0.0, gain) * TAX_RATE
        total_gain += gain
        total_tax += tax
        rows.append({
            "isin": isin,
            "name": row.get("Name", isin),
            "avg_cost": avg_cost,
            "current_price": current_price,
            "quantity": qty,
            "unrealized_gain": round(gain, 2),
            "tax_estimate": round(tax, 2),
        })
    return jsonify({
        "holdings": rows,
        "total_unrealized_gain": round(total_gain, 2),
        "total_tax_estimate": round(total_tax, 2),
        "note": "Vereinfacht: kein Freistellungsauftrag, keine Teilverkäufe berücksichtigt",
    })


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 404
    body = request.get_json(force=True) or {}
    modifications = body.get("modifications", [])  # [{isin, amount_eur}]

    net_values = {row["ISIN"]: float(row["netValue"]) for _, row in df.iterrows()}
    names = {row["ISIN"]: row.get("Name", row["ISIN"]) for _, row in df.iterrows()}

    for mod in modifications:
        isin = mod.get("isin", "")
        delta = float(mod.get("amount_eur", 0))
        net_values[isin] = max(0.0, net_values.get(isin, 0) + delta)

    new_total = sum(net_values.values())
    if new_total <= 0:
        return jsonify({"error": "Portfolio value would be zero"}), 422
    new_weights = {k: v / new_total for k, v in net_values.items() if v > 0}

    all_tickers = list(new_weights.keys())
    try:
        prices = _fetch_prices(all_tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    prices = prices.reindex(columns=all_tickers).dropna()
    if prices.empty:
        return jsonify({"error": "no_price_data"}), 422

    orig_total = float(df["netValue"].sum())
    orig_weights = {row["ISIN"]: float(row["netValue"]) / orig_total for _, row in df.iterrows()}
    try:
        orig_perf = compute_performance(prices, orig_weights)
        new_perf = compute_performance(prices, new_weights)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    return jsonify({
        "original": {
            "weights": orig_weights,
            "sharpe": orig_perf["sharpe"],
            "volatility": orig_perf["volatility"],
            "cagr": orig_perf["cagr"],
        },
        "modified": {
            "weights": new_weights,
            "sharpe": new_perf["sharpe"],
            "volatility": new_perf["volatility"],
            "cagr": new_perf["cagr"],
        },
        "delta_sharpe": new_perf["sharpe"] - orig_perf["sharpe"],
        "names": names,
    })


@app.route("/api/overlap")
def api_overlap():
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 422
    tickers = df["ISIN"].tolist()
    if len(tickers) < 2:
        return jsonify({"tickers": tickers, "pairs": [], "diversification_score": 1.0})
    try:
        prices = _fetch_prices(tickers)
    except Exception as e:
        return jsonify({"error": str(e)}), 422
    result = compute_etf_overlap(prices)
    names = {row["ISIN"]: row.get("Name", row["ISIN"]) for _, row in df.iterrows()}
    for pair in result["pairs"]:
        pair["name_a"] = names.get(pair["a"], pair["a"])
        pair["name_b"] = names.get(pair["b"], pair["b"])
    return jsonify(result)


@app.route("/api/buy-recommendation", methods=["POST"])
def api_buy_recommendation():
    data = request.get_json() or {}
    target_weights = data.get("target_weights", {})
    try:
        invest_amount = float(data.get("invest_amount", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invest_amount must be numeric"}), 422
    if not target_weights:
        return jsonify({"error": "target_weights required"}), 422
    if invest_amount <= 0:
        return jsonify({"error": "invest_amount must be > 0"}), 422
    df = _load_portfolio()
    if df is None:
        return jsonify({"error": "no_portfolio"}), 422
    current_net_values = {row["ISIN"]: float(row["netValue"]) for _, row in df.iterrows()}
    names = {row["ISIN"]: row.get("Name", row["ISIN"]) for _, row in df.iterrows()}
    result = compute_buy_recommendation(current_net_values, target_weights, invest_amount)
    if "error" in result:
        return jsonify(result), 422
    result["names"] = names
    return jsonify(result)


@app.route("/api/insights")
def api_insights():
    df = _load_portfolio()
    if df is None:
        return jsonify({"insights": []})
    insights = []
    total = float(df["netValue"].sum())
    if total <= 0:
        return jsonify({"insights": []})
    weights = {row["ISIN"]: float(row["netValue"]) / total for _, row in df.iterrows()}
    names = {row["ISIN"]: row.get("Name", row["ISIN"]) for _, row in df.iterrows()}

    # Herfindahl concentration index
    hhi = sum(w ** 2 for w in weights.values())
    if hhi > 0.35:
        top = max(weights, key=weights.get)
        insights.append({
            "type": "concentration", "severity": "warning",
            "message": f"Konzentration hoch (HHI {hhi:.2f}) — {names.get(top, top)} dominiert.",
            "action": "Optimizer → Min Volatilität für bessere Streuung.",
        })

    # Single position > 30%
    for isin, w in weights.items():
        if w > 0.30:
            insights.append({
                "type": "overweight", "severity": "warning",
                "message": f"{names.get(isin, isin)}: {w*100:.1f}% — Einzelposition über 30%.",
                "action": "Rebalancing oder Optimizer nutzen.",
            })

    # Too few positions
    if len(df) < 3:
        insights.append({
            "type": "diversification", "severity": "info",
            "message": f"Nur {len(df)} Position(en) — Diversifikation sehr begrenzt.",
            "action": "Weitere ETFs oder Assetklassen hinzufügen.",
        })

    # Overlap check
    tickers = df["ISIN"].tolist()
    if len(tickers) >= 2:
        try:
            prices = _fetch_prices(tickers)
            overlap = compute_etf_overlap(prices)
            for pair in overlap["pairs"]:
                if pair["level"] == "high":
                    na = names.get(pair["a"], pair["a"])
                    nb = names.get(pair["b"], pair["b"])
                    insights.append({
                        "type": "overlap", "severity": "warning",
                        "message": f"Hohe Überschneidung: {na} ↔ {nb} (Korr. {pair['correlation']:.2f})",
                        "action": "Allokation-Tab → Overlap für Details.",
                    })
        except Exception:
            pass

    if not insights:
        insights.append({
            "type": "ok", "severity": "ok",
            "message": "Keine kritischen Auffälligkeiten — Portfolio sieht solide aus.",
            "action": "",
        })
    return jsonify({"insights": insights})


if __name__ == "__main__":
    app.run(debug=True)
