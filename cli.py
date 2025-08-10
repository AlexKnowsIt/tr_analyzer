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
    tr.login()
    pf = Portfolio(tr)
    asyncio.run(pf.portfolio_loop())
    output_path = Path(output)
    pf.portfolio_to_csv(output_path)
    return output_path


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
