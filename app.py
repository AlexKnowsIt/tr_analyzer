"""Simple web application for portfolio analysis."""

import base64
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import yfinance as yf
from flask import Flask, render_template, request
from matplotlib.figure import Figure
from pypfopt import EfficientFrontier, expected_returns, risk_models, plotting
from optimizer import parse_constraints

app = Flask(__name__)
PORTFOLIO_FILE = Path("portfolio.csv")


@app.route("/", methods=["GET", "POST"])
def index():
    if not PORTFOLIO_FILE.exists():
        return "Portfolio file not found. Run the CLI to download your portfolio first."

    df = pd.read_csv(PORTFOLIO_FILE, sep=";")
    tickers = df["ISIN"].tolist()
    constraint_str = request.form.get("constraints", "")

    info_rows = []
    for isin in tickers:
        ticker = yf.Ticker(isin)
        info = ticker.info
        quote_type = info.get("quoteType", "")
        info_rows.append(
            {
                "ISIN": isin,
                "region": info.get("country", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "asset_type": "ETF" if quote_type == "ETF" else "Stock",
            }
        )
    info_df = pd.DataFrame(info_rows)
    merged = df.merge(info_df, on="ISIN", how="left")

    fig_sector = px.sunburst(
        merged,
        path=["asset_type", "sector", "Name"],
        values="netValue",
        title="Allocation by Sector",
    )
    fig_country = px.sunburst(
        merged,
        path=["asset_type", "region", "Name"],
        values="netValue",
        title="Allocation by Country",
    )
    sunburst_sector_html = fig_sector.to_html(full_html=False, include_plotlyjs="cdn")
    sunburst_country_html = fig_country.to_html(full_html=False, include_plotlyjs=False)

    price_data = yf.download(tickers=tickers, period="5y")['Adj Close'].dropna()
    if price_data.empty or price_data.isnull().all().all():
        ef_html = "<p>No price data available for efficient frontier.</p>"
        weights_html = ""
        perf_html = ""
    else:
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        ef = EfficientFrontier(mu, S)

        for idx, op, val in parse_constraints(constraint_str, tickers):
            if op == "<=":
                ef.add_constraint(lambda w, idx=idx, val=val: w[idx] <= val)
            else:
                ef.add_constraint(lambda w, idx=idx, val=val: w[idx] >= val)

        fig = Figure()
        ax = fig.subplots()
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        ef_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}"/>'

        ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        weights_df = (
            pd.Series(cleaned_weights)
            .reset_index()
            .rename(columns={"index": "ISIN", 0: "weight"})
        )
        weights_html = weights_df.to_html(index=False, float_format="{:.2%}".format)
        perf = ef.portfolio_performance()
        perf_html = (
            f"<p>Return: {perf[0]:.2%}<br>Risk: {perf[1]:.2%}<br>Sharpe: {perf[2]:.2f}</p>"
        )

    return render_template(
        "index.html",
        sunburst_sector=sunburst_sector_html,
        sunburst_country=sunburst_country_html,
        ef_plot=ef_html,
        weights_table=weights_html,
        perf_html=perf_html,
        constraints=constraint_str,
    )


if __name__ == "__main__":
    app.run(debug=True)
