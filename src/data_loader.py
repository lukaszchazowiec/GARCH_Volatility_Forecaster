import pandas as pd
import yfinance as yf
import os

TICKER = "^GSPC"
START_DATE = "2009-01-01"
CACHE_PATH = "../data/sp500_prices.csv"

def fetch_prices(force_download=False):
    if os.path.exists(CACHE_PATH) and not force_download:
        print(f"Loading data from cache: {CACHE_PATH}")
        prices = pd.read_csv(CACHE_PATH, index_col="Date", parse_dates=True)
        return prices

    # download from Yahoo Finance
    print(f"Downloading {TICKER} from Yahoo Finance...")
    raw = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)

    prices = pd.DataFrame()
    prices["Close"] = raw["Close"].squeeze()
    prices = prices.dropna()

    os.makedirs("../data", exist_ok=True)
    prices.to_csv(CACHE_PATH)
    print(f"Saved to {CACHE_PATH}")

    return prices


def get_close(force_download=False):
    prices = fetch_prices(force_download)
    return prices["Close"]


if __name__ == "__main__":
    close = get_close()

    print(f"\nTicker       : {TICKER}")
    print(f"Start        : {close.index[0].date()}")
    print(f"End          : {close.index[-1].date()}")
    print(f"Observations : {len(close)}")
    print(f"Missing      : {close.isna().sum()}")
    print(f"\nFirst 3 rows:\n{close.head(3)}")
    print(f"\nLast 3 rows:\n{close.tail(3)}")