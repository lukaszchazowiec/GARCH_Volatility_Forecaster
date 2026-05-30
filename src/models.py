from data_loader import get_close
from returns import compute_returns
import pandas as pd
import numpy as np
from arch import arch_model


# Fit GARCH

def fit_garch(returns, label=""):
    model = arch_model(returns, p=1, q=1, dist='t')
    results = model.fit(disp='off')

    alpha = results.params['alpha[1]']
    beta  = results.params['beta[1]']

    print(f"GARCH(1,1) Results ({label})")
    print(f"omega              : {results.params['omega']:.6f}")
    print(f"alpha[1]           : {alpha:.4f}")
    print(f"beta[1]            : {beta:.4f}")
    print(f"persistence        : {alpha + beta:.4f}")
    print(f"nu (df)            : {results.params['nu']:.4f}")
    print(f"log-likelihood     : {results.loglikelihood:.4f}")
    print(f"AIC                : {results.aic:.4f}")
    print(f"BIC                : {results.bic:.4f}")

    return {
        'label'      : label,
        'model'      : 'GARCH(1,1)',
        'omega'      : results.params['omega'],
        'alpha'      : alpha,
        'beta'       : beta,
        'persistence': alpha + beta,
        'nu'         : results.params['nu'],
        'loglik'     : results.loglikelihood,
        'aic'        : results.aic,
        'bic'        : results.bic,
        'cond_vol'   : results.conditional_volatility,
        'results_obj': results
    }

# Fit EGARCH







# Quick check
if __name__ == "__main__":
    close = get_close()
    returns = compute_returns(close)
    results = fit_garch(returns, label="S&P 500")