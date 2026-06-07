from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot
from scipy import stats
import matplotlib.pyplot as plt
from data_loader import get_close
from returns import compute_returns


# Pre-fit functions

def arch_lm_test(returns, label="", post_fit=False):
    test_stat, p_value, f_stat, f_pvalue = het_arch(returns)
    print(f"ARCH-LM Test ({label})")
    print(f"ARCH-LM Test Statistic: {test_stat:.4f}")
    print(f"ARCH-LM Test p-value: {p_value:.4f}")

    if post_fit:
        if p_value < 0.05:
            print("Conclusion: ARCH effects remain in residuals - model may be misspecified.")
        else:
            print("Conclusion: no ARCH effects in residuals - GARCH successfully captured volatility dynamics.")
    else:
        if p_value < 0.05:
            print("Conclusion: ARCH effects present - GARCH modeling is justified.")
        else:
            print("Conclusion: no ARCH effects present - GARCH modeling is not justified.")


def plot_acf_pacf(returns):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ACF and PACF of Returns and Squared Returns")

    plot_acf(returns, ax=axes[0, 0], title="ACF of Returns")
    plot_pacf(returns, ax=axes[0, 1], title="PACF of Returns")
    plot_acf(returns**2, ax=axes[1, 0], title="ACF of Squared Returns")
    plot_pacf(returns**2, ax=axes[1, 1], title="PACF of Squared Returns")

    plt.tight_layout()
    plt.show()



# Post-fit functions

def ljung_box_test(residuals, label=""):
    ljung_box_test_result = acorr_ljungbox(residuals, lags=10, return_df=True)
    print(f"Ljung-Box Test ({label})")
    print(f"H0: no autocorrelation in residuals | tested at lags 1-10")
    print("-" * 45)
    print(ljung_box_test_result.to_string())
    print("-" * 45)
    if (ljung_box_test_result["lb_pvalue"] < 0.05).any():
        print("Conclusion: autocorrelation remains in residuals - model may be misspecified.")
    else:
        print("Conclusion: no autocorrelation in residuals - GARCH captured the dynamics.")



def plot_qq(residuals, label="", df=None):
    plt.figure(figsize=(8, 6))

    if df is not None:
        probplot(residuals, dist=stats.t, sparams=(df,), plot=plt)
        plt.title(f"QQ Plot of Standardized Residuals vs Student's t (df={df:.2f}) ({label})")
    else:
        probplot(residuals, dist="norm", plot=plt)
        plt.title(f"QQ Plot of Standardized Residuals vs Normal ({label})")

    plt.grid(True)
    plt.show()


# Quick check
if __name__ == "__main__":
    from models import fit_garch, fit_egarch, fit_gjr, compare_models
    close = get_close()
    returns = compute_returns(close)

    print("Initial Diagnostics:")
    arch_lm_test(returns, label="raw returns")
    plot_acf_pacf(returns)

    print("\nFitting Models:")
    garch_res = fit_garch(returns, label="S&P 500")
    egarch_res = fit_egarch(returns, label="S&P 500")
    gjr_res = fit_gjr(returns, label="S&P 500")

    print("\nComparison of Models:")
    model_results_list = [garch_res, egarch_res, gjr_res]
    compare_models(model_results_list)

    print("\nPost-Fit Diagnostics:")
    for res in model_results_list:
        model_name = res['model']
        results_obj = res['results_obj']

        std_residuals = results_obj.resid / results_obj.conditional_volatility

        print("\n" + "=" * 40)
        print(f"Diagnostics for {model_name} model:")
        print("=" * 40)

        ljung_box_test(std_residuals, label=model_name)

        estimated_df = results_obj.params['nu']

        plot_qq(std_residuals, label=model_name, df=estimated_df)