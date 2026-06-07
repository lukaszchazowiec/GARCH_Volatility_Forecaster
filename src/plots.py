import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.stats import norm

from data_loader import get_close
from returns import compute_returns
from models import fit_garch, fit_egarch, fit_gjr
from backtest import rolling_var_backtest

# plot_conditional_volatility(all_results, returns)
# plot_var_backtest(actual, var_95, var_99, model_name)
# plot_return_distribution(returns)
# plot_qq(residuals, label="")  ← already written in diagnostics.py

def plot_returns(returns):

    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=100, density=True, alpha=0.6, color="steelblue", label="Actual Returns")
    plt.title("S&P 500 Daily Log Returns vs Normal Distribution")
    plt.xlabel("Returns")
    plt.ylabel("Probability")
    plt.legend()

    x = np.linspace(returns.min(), returns.max(), 200)
    normal_curve = norm.pdf(x, returns.mean(), returns.std())
    plt.plot(x, normal_curve, color='red', linewidth=2, label="Normal Distribution")
    plt.show()



def plot_conditional_volatility(garch_results, egarch_results, gjr_garch_results):

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor('#f5f5f5')
    plt.gca().set_facecolor('#ebebeb')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    garch_cond_vol = garch_results['cond_vol'] / 100
    egarch_cond_vol = egarch_results['cond_vol'] / 100
    gjr_garch_cond_vol = gjr_garch_results['cond_vol'] / 100

    plt.plot(garch_cond_vol, color='blue', linewidth=1.5, alpha=0.6)
    plt.plot(egarch_cond_vol, color='red', linewidth=1.5, alpha=0.6)
    plt.plot(gjr_garch_cond_vol, color='green', linewidth=1.5, alpha=0.6)
    plt.legend(["GARCH", "EGARCH", "GJR"])
    plt.title("Timeline of Conditional Volatility")
    plt.xlabel("Timeline")
    plt.ylabel("Conditional Volatility")
    plt.show()


def plot_var_backtest(actual_returns, var_95, var_99, model_name):

    exceedances = actual_returns[actual_returns < -var_95]

    plt.figure(figsize=(14, 6))

    plt.plot(actual_returns, color='grey', linewidth=0.7, alpha=0.7, label='Actual Returns')
    plt.plot(-var_95, color='blue', linewidth=1, label='VaR 95%')
    plt.plot(-var_99, color='red', linewidth=1, label='VaR 99%')
    plt.scatter(exceedances.index, exceedances.values, color='orange', s=10, zorder=5, label='Exceedances')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x * 100:.1f}%'))
    plt.title(f"VaR Backtest — {model_name}")
    plt.legend(["Actual Returns", "VaR 95%", "VaR 99%", "Exceedances"])
    plt.show()




if __name__ == "__main__":

    close = get_close()
    returns = compute_returns(close)
    plot_returns(returns)


    # Plot Conditional Volatility
    garch_results = fit_garch(returns, label='S&P 500')
    egarch_results = fit_egarch(returns, label='S&P 500')
    gjr_garch_results = fit_gjr(returns, label='S&P 500')
    plot_conditional_volatility(garch_results, egarch_results, gjr_garch_results)


    # Plot VaR Backtest
    actual_returns = pd.read_csv("../data/actual_returns.csv", index_col=0, parse_dates=True).squeeze()

    # GARCH
    var_95_garch = pd.read_csv("../data/var_95_GARCH.csv", index_col=0, parse_dates=True).squeeze()
    var_99_garch = pd.read_csv("../data/var_99_GARCH.csv", index_col=0, parse_dates=True).squeeze()

    # E-GARCH
    var_95_egarch = pd.read_csv("../data/var_95_EGARCH.csv", index_col=0, parse_dates=True).squeeze()
    var_99_egarch = pd.read_csv("../data/var_99_EGARCH.csv", index_col=0, parse_dates=True).squeeze()

    # GJR-GARCH
    var_95_gjr = pd.read_csv("../data/var_95_GJR-GARCH.csv", index_col=0, parse_dates=True).squeeze()
    var_99_gjr = pd.read_csv("../data/var_99_GJR-GARCH.csv", index_col=0, parse_dates=True).squeeze()

    plot_var_backtest(actual_returns, var_95_garch, var_99_garch, "GARCH")
    plot_var_backtest(actual_returns, var_95_egarch, var_99_egarch, "EGARCH")
    plot_var_backtest(actual_returns, var_95_gjr, var_99_gjr, "GJR-GARCH")