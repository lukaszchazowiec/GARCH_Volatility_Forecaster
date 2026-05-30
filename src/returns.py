from data_loader import get_close
import numpy as np
import scipy.stats as stats


# Because of the common use of log returns, I will refer to log returns simply as 'returns'
def compute_returns(close):
    returns = np.log(close/close.shift(1)).dropna()
    return returns


def describe_returns(returns):
    summary = returns.describe()
    skewness = returns.skew()
    kurtosis = returns.kurt()
    print(summary)
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")


def jarque_bera_test(returns):
    """
    jarque-bera test for normality;
    returns statistic and p-value;
    p-value < 0,05- reject null hypothesis that returns are normally distributed
    """
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    print(f"Jarque-Bera Test: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
    if jb_pvalue < 0.05:
        print("Conclusion: returns are not normally distributed - GARCH modeling is justified.")
    else:
        print("Conclusion: cannot reject normality.")


# Quick check
if __name__ == "__main__":
    close = get_close()
    returns = compute_returns(close)
    describe_returns(returns)
    jarque_bera_test(returns)