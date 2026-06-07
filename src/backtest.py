import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t
from tqdm import tqdm
from scipy.stats import chi2

from data_loader import get_close
from returns import compute_returns


def rolling_var_backtest(returns, model_type, window=252):
    actual_returns = []
    var_95 = []
    var_99 = []
    for i in tqdm(range(window, len(returns))):

        window_data = returns.iloc[i-window:i]
        actual_returns.append(returns.iloc[i])

        if model_type == 'GARCH':
            model = arch_model(window_data * 100, p=1, q=1, vol='GARCH', dist='t')

        elif model_type == 'EGARCH':
            model = arch_model(window_data * 100, p=1, q=1, vol='EGARCH', dist='t')

        elif model_type == 'GJR-GARCH':
            model = arch_model(window_data * 100, p=1, q=1, o=1, vol='GARCH', dist='t')


        results = model.fit(disp='off', show_warning=False)

        pred_variance = results.forecast(horizon=1).variance.iloc[-1, 0]
        sigma_next = np.sqrt(pred_variance) / 100
        nu = results.params['nu']

        v95 = -t.ppf(0.05, nu) * sigma_next
        v99 = -t.ppf(0.01, nu) * sigma_next

        var_95.append(v95)
        var_99.append(v99)

    index = returns.index[window:]


    return (
        pd.Series(actual_returns, index=index),
        pd.Series(var_95, index=index),
        pd.Series(var_99, index=index)
    )


def statistical_tests(actual_returns, var_forecasts, alpha):
    actual_returns = np.array(actual_returns)
    var_forecasts = np.array(var_forecasts)

    # --- Basic statistics ---
    hits = np.where(actual_returns < -var_forecasts, 1, 0)
    N = len(hits)
    x = sum(hits)
    p_actual = x / N

    print(f"Number of days in the test: {N}")
    print(f"Number of var breaches: {x}")

    def safe_log(val):
        return np.log(val) if val > 0 else 0.0


    # --- Kupiec POF Test ---
    L_null = (N-x) * np.log(1-alpha) + x * np.log(alpha)
    L_alt = (N-x) * np.log(1-p_actual) + x * np.log(p_actual)

    LR_uc = 2 * (L_alt - L_null)
    p_value = chi2.sf(LR_uc, df=1)


    # --- Christoffersen Independence Test ---
    n00, n01, n10, n11 = 0, 0, 0, 0

    for i in range(1, len(hits)):

        if hits[i-1] == 0 and hits[i] == 0:
            n00 += 1
        elif hits[i-1] == 0 and hits[i] == 1:
            n01 += 1
        elif hits[i-1] == 1 and hits[i] == 0:
            n10 += 1
        elif hits[i-1] == 1 and hits[i] == 1:
            n11 += 1


    p_01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0.0
    p_11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0.0
    p_total = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0.0

    L_ind_null = (n00+n10) * safe_log(1-p_total) + (n01+n11) * safe_log(p_total)
    L_ind_alt = n00 * safe_log(1-p_01) + n01 * safe_log(p_01) + n10 * safe_log(1-p_11) + n11 * safe_log(p_11)

    LR_ind = 2 * (L_ind_alt - L_ind_null)
    p_value_ind = chi2.sf(LR_ind, df=1)


    # --- Basel III Test ---
    basel_3_test = "N/A"
    if alpha == 0.01:

        scaled_exceptions = (x/N) * 250

        if scaled_exceptions <= 4:
            basel_3_test = "GREEN"
        elif scaled_exceptions <= 9:
            basel_3_test = "YELLOW"
        else:
            basel_3_test = "RED"


    return {
        "hits": hits,
        "p_actual": p_actual,
        "p_value": p_value,
        "p_value_ind": p_value_ind,
        "basel_3_test": basel_3_test
    }



# Quick check

if __name__ == "__main__":
    # 1. Fetch and prepare market data
    close = get_close()
    returns = compute_returns(close)

    # List of volatility models to backtest
    list_of_models = ['GARCH', 'EGARCH', 'GJR-GARCH']

    # Dictionary to store full results and prevent data loss after the loop
    all_results = {}

    # 2. Main computation loop (GARCH Volatility Engine)
    for model_type in list_of_models:
        print(f"\n=========================================")
        print(f" Running backtest for model: {model_type}")
        print(f"=========================================")

        # Execute rolling window calculations
        actual, v95, v99 = rolling_var_backtest(returns, model_type=model_type, window=252)

        # Store generated series in the dictionary
        all_results[model_type] = {
            'actual': actual,
            'var_95': v95,
            'var_99': v99
        }

    for model_type in list_of_models:
        res = all_results[model_type]
        res['actual'].to_csv(f"../data/actual_returns.csv")
        res['var_95'].to_csv(f"../data/var_95_{model_type}.csv")
        res['var_99'].to_csv(f"../data/var_99_{model_type}.csv")


    # 3. Generate the final evaluation report
    print("\n\n" + "=" * 60)
    print("  FINAL REPORT - STATISTICAL TESTS (Christoffersen 1998)")
    print("=" * 60)

    for model_type in list_of_models:
        res = all_results[model_type]

        # Run your universal testing function for both significance levels
        stats_95 = statistical_tests(res['actual'], res['var_95'], alpha=0.05)
        stats_99 = statistical_tests(res['actual'], res['var_99'], alpha=0.01)

        print(f"\nRESULTS FOR MODEL: {model_type}")
        print(f"  " + "-" * 40)

        # VaR 95% evaluation section
        print(f"  [VaR 95%] Actual failure rate: {stats_95['p_actual'] * 100:.2f}% (Expected: 5.00%)")
        print(f"            Kupiec POF p-value: {stats_95['p_value']:.4f} " + (
            "REJECT" if stats_95['p_value'] < 0.05 else "PASSED"))
        print(f"            Christoffersen Ind p-value: {stats_95['p_value_ind']:.4f} " + (
            "CLUSTERING DETECTED" if stats_95['p_value_ind'] < 0.05 else "PASSED"))

        # VaR 99% evaluation section
        print(f"  [VaR 99%] Actual failure rate: {stats_99['p_actual'] * 100:.2f}% (Expected: 1.00%)")
        print(f"            Kupiec POF p-value: {stats_99['p_value']:.4f} " + (
            "REJECT" if stats_99['p_value'] < 0.05 else "PASSED"))
        print(f"            Christoffersen Ind p-value: {stats_99['p_value_ind']:.4f} " + (
            "CLUSTERING DETECTED" if stats_99['p_value_ind'] < 0.05 else "PASSED"))
        print(f"            Basel III Zone: {stats_99['basel_3_test']}")
        print("  " + "-" * 40)