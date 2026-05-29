from data_loader import get_close
import numpy as np


# Because of the common use of log returns, I will refer to lo returns simply as 'returns'
def compute_returns(close):
    returns = np.log(close/close.shift(1)).dropna()
    return returns


def describe_returns(returns):
    stats = returns.describe()
    skewness = returns.skew()
    kurtosis = returns.kurt()
    print(stats)
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")


# Quick check
if __name__ == "__main__":
    close = get_close()
    returns = compute_returns(close)
    describe_returns(returns)