#!/usr/bin/env python3
"""Command line utilities for TradeRepublic portfolio download."""

import argparse
import asyncio
from pathlib import Path

from pytr.api import TradeRepublicApi
from pytr.portfolio import Portfolio


def download_portfolio(phone: str, pin: str, output: str = "portfolio.csv") -> Path:
    """Download the current TradeRepublic portfolio.

    Parameters
    ----------
    phone: str
        Phone number used for TradeRepublic login.
    pin: str
        Trading PIN.
    output: str
        Path to the CSV file that will be written.
    """
    tr = TradeRepublicApi(phone_no=phone, pin=pin)
    countdown = tr.initiate_weblogin()
    code = input(f"2FA-Code (läuft in {countdown}s ab): ").strip()
    tr.complete_weblogin(code)
    pf = Portfolio(tr, output=output)
    asyncio.run(pf.portfolio_loop())
    pf.portfolio_to_csv()
    return Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TradeRepublic portfolio")
    parser.add_argument("phone", help="Phone number used for TradeRepublic login")
    parser.add_argument("pin", help="Trading PIN")
    parser.add_argument(
        "--output", "-o", default="portfolio.csv", help="Output CSV file path"
    )
    args = parser.parse_args()
    download_portfolio(args.phone, args.pin, args.output)


if __name__ == "__main__":
    main()
