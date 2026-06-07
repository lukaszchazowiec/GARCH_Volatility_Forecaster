from data_loader import get_close
from returns import compute_returns
import pandas as pd
from arch import arch_model


# Fit GARCH

def fit_garch(returns, label=""):
    model = arch_model(returns*100, p=1, q=1, dist='t')
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
    print()

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

def fit_egarch(returns, label=""):
    model = arch_model(returns*100, p=1, q=1, vol='EGARCH', dist='t')
    results = model.fit(disp='off')

    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']

    print(f"EGARCH(1,1) Results ({label})")
    print(f"omega              : {results.params['omega']:.6f}")
    print(f"alpha[1]           : {alpha:.4f}")
    print(f"beta[1]            : {beta:.4f}")
    print(f"persistence (beta) : {beta:.4f}")
    print(f"nu (df)            : {results.params['nu']:.4f}")
    print(f"log-likelihood     : {results.loglikelihood:.4f}")
    print(f"AIC                : {results.aic:.4f}")
    print(f"BIC                : {results.bic:.4f}")
    print()

    return {
        'label': label,
        'model': 'EGARCH(1,1)',
        'omega': results.params['omega'],
        'alpha': alpha,
        'beta': beta,
        'persistence': beta,
        'nu': results.params['nu'],
        'loglik': results.loglikelihood,
        'aic': results.aic,
        'bic': results.bic,
        'cond_vol': results.conditional_volatility,
        'results_obj': results
    }


# Fit GJR-GARCH

def fit_gjr(returns, label=""):
    model = arch_model(returns*100, p=1, q=1, o=1, vol='GARCH', dist='t')
    results = model.fit(disp='off')

    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']
    gamma = results.params['gamma[1]']
    persistence = alpha + beta + gamma/2

    print(f"GJR-GARCH(1,1) Results ({label})")
    print(f"omega              : {results.params['omega']:.6f}")
    print(f"alpha[1]           : {alpha:.4f}")
    print(f"beta[1]            : {beta:.4f}")
    print(f"gamma              : {gamma:.4f}")
    print(f"persistence        : {persistence:.4f}")
    print(f"nu (df)            : {results.params['nu']:.4f}")
    print(f"log-likelihood     : {results.loglikelihood:.4f}")
    print(f"AIC                : {results.aic:.4f}")
    print(f"BIC                : {results.bic:.4f}")
    print()

    return {
        'label': label,
        'model': 'GJR-GARCH(1,1)',
        'omega': results.params['omega'],
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'persistence': alpha + gamma / 2 + beta,
        'nu': results.params['nu'],
        'loglik': results.loglikelihood,
        'aic': results.aic,
        'bic': results.bic,
        'cond_vol': results.conditional_volatility,
        'results_obj': results
    }


# Compare models

def compare_models(model_results_list):
    rows=[]
    for r in model_results_list:
        rows.append({
            'Model': r['model'],
            'Alpha': round(r['alpha'], 4),
            'Beta': round(r['beta'], 4),
            'Gamma': round(r.get('gamma', float('nan')), 4),
            'Persistence': round(r['persistence'], 4),
            'Nu': round(r['nu'], 4),
            'Log-Lik': round(r['loglik'], 2),
            'AIC': round(r['aic'], 2),
            'BIC': round(r['bic'], 2),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# Quick check
if __name__ == "__main__":
    close = get_close()
    returns = compute_returns(close)
    garch_results = fit_garch(returns, label="S&P 500")
    egarch_results = fit_egarch(returns, label="S&P 500")
    gjr_results = fit_gjr(returns, label="S&P 500")

    model_results_list = [garch_results, egarch_results, gjr_results]
    compare_models(model_results_list)