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
    compute_frontier,
    compute_performance,
    parse_constraints,
    run_optimization,
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


if __name__ == "__main__":
    app.run(debug=True)
