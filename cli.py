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
    if not hasattr(tr, "sk"):
        raise RuntimeError(
            "Signing key not found. Pair your device with TradeRepublic to generate the key file."
        )
    tr.login()
    pf = Portfolio(tr)
    asyncio.run(pf.portfolio_loop())
    output_path = Path(output)
    pf.portfolio_to_csv(output_path)
    return output_path


def pair_device(phone: str, pin: str) -> None:
    """Generate and store a signing key by pairing the device.

    The TradeRepublic backend sends a confirmation code to the registered
    device. Enter the code when prompted and the signing key will be written to
    the default location used by ``pytr`` (``~/.pytr/keyfile.pem``).
    """
    tr = TradeRepublicApi(phone_no=phone, pin=pin)
    tr.initiate_device_reset()
    token = input("Enter the pairing code sent by TradeRepublic: ")
    tr.complete_device_reset(token)


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeRepublic utilities")
    parser.add_argument("phone", help="Phone number used for TradeRepublic login")
    parser.add_argument("pin", help="Trading PIN")
    parser.add_argument(
        "--output", "-o", default="portfolio.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="Pair device and create signing key instead of downloading portfolio",
    )
    args = parser.parse_args()
    if args.pair:
        pair_device(args.phone, args.pin)
    else:
        download_portfolio(args.phone, args.pin, args.output)


if __name__ == "__main__":
    main()
