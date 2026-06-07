# Volatility Forecasting with GARCH: S&P 500


A rigorous empirical study of volatility dynamics in the U.S. equity market, using the S&P 500 index (^GSPC) as the primary data series. The project fits and compares three GARCH-family models — GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) — validates them through residual diagnostics, and evaluates their practical usefulness via rolling Value-at-Risk backtests with formal statistical tests.

This project was motivated by work in a student investment fund risk department, where GARCH-based volatility forecasting underpins VaR estimation for portfolio risk monitoring. The S&P 500 was selected as a liquid, well-documented benchmark with reliable data availability.

---

## Motivation

Daily equity returns are well-documented to exhibit two stylized facts that violate the assumptions of constant-variance models: **volatility clustering** (large shocks tend to be followed by large shocks) and **fat tails** (extreme returns occur far more frequently than a normal distribution predicts). These properties, first formally characterised by Engle (1982) and extended by Bollerslev (1986), motivate the use of autoregressive conditional heteroscedasticity models.

The S&P 500 is one of the most studied equity indices in financial econometrics, making it an ideal benchmark for validating GARCH methodology. Its long history, liquidity, and data availability make results reproducible and comparable to the existing academic literature.

The central research question is: **do GARCH-family models produce statistically valid one-day-ahead VaR forecasts for the S&P 500, and does the choice of GARCH specification matter?**

---

## Project Structure

```
GARCH_Volatility_Forecaster/
│
├── README.md
├── requirements.txt
├── data/                       # raw price data (gitignored, fetched via yfinance)
├── src/
│   ├── data_loader.py          # price fetching and caching
│   ├── returns.py              # log return computation and descriptive statistics
│   ├── diagnostics.py          # pre- and post-fit statistical tests
│   ├── models.py               # GARCH, EGARCH, GJR-GARCH fitting and selection
│   ├── backtest.py             # rolling VaR backtest and Kupiec/Christoffersen tests
│   └── plots.py                # all visualisations: conditional vol, VaR backtest, distributions
└── notebooks/
    └── analysis.ipynb          # full narrative analysis with plots and results tables
```

---

## Methodology

### 1. Data

Daily closing prices for the S&P 500 (`^GSPC`) are sourced via `yfinance` from 2005-01-01 to present, yielding approximately 5,100 observations. Starting from 2005 captures the full 2008 financial crisis and multiple volatility regimes including the European debt crisis (2011), COVID-19 crash (2020), and the 2022 rate hike cycle. Log returns are computed as $r_t = \ln(P_t / P_{t-1})$.

### 2. Pre-fit Diagnostics — Motivating GARCH

Before fitting any model, the presence of ARCH effects is formally verified:

- **Jarque-Bera test** on raw returns to confirm non-normality and fat tails
- **ACF/PACF plots** of returns and squared returns — autocorrelation in squared returns is the defining signature of volatility clustering
- **ARCH-LM test** (Engle, 1982) — null hypothesis of no ARCH effects; rejection motivates the entire modelling exercise

This step is not a formality. If ARCH effects were absent, GARCH would be an unnecessary complication. Documenting this evidence is the analytical foundation of the project.

### 3. Model Specifications

Three GARCH-family models are estimated via Maximum Likelihood with Student-t distributed errors, chosen over Gaussian based on the excess kurtosis (11.70) observed in the return series and consistent with Hall & Yao (2003):

**GARCH(1,1)** — Bollerslev (1986)

$$\sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

The persistence parameter $\alpha + \beta$ measures how long volatility shocks decay. Values close to 1 indicate long memory in volatility, commonly observed in equity markets.

**EGARCH(1,1)** — Nelson (1991)

$$\ln(\sigma^2_t) = \omega + \alpha \left( |z_{t-1}| - \mathbb{E}|z_{t-1}| \right) + \gamma z_{t-1} + \beta \ln(\sigma^2_{t-1})$$

The log-variance formulation guarantees positive variance without parameter restrictions. The $\gamma$ coefficient captures the **leverage effect**: negative shocks ($z_{t-1} < 0$) increase volatility by more than positive shocks of equal magnitude.

**GJR-GARCH(1,1)** — Glosten, Jagannathan & Runkle (1993)

$$\sigma^2_t = \omega + (\alpha + \gamma \mathbf{1}_{[\varepsilon_{t-1}<0]}) \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}$$

An alternative leverage-effect specification. The indicator function $\mathbf{1}_{[\varepsilon_{t-1}<0]}$ adds an asymmetric component $\gamma$ only when the previous shock was negative. A statistically significant $\gamma > 0$ confirms the leverage effect.

Models are compared using AIC, BIC, and log-likelihood. Both information criteria penalise model complexity, making them appropriate for comparing specifications with different numbers of parameters.

### 4. Post-fit Residual Diagnostics

A well-specified GARCH model should produce standardised residuals $\hat{z}_t = \varepsilon_t / \hat{\sigma}_t$ that behave as white noise. This is verified via:

- **Ljung-Box test** on $\hat{z}_t$ — tests for remaining serial autocorrelation
- **ARCH-LM test** on residuals — confirms that ARCH effects have been absorbed by the model
- **QQ plot** of standardised residuals against the fitted Student-t distribution — assesses whether the distributional assumption is appropriate

The before/after ARCH-LM comparison is the key diagnostic narrative: significant pre-fit, insignificant post-fit confirms the model captured volatility dynamics.

### 5. VaR Backtesting

A rolling window backtest evaluates the practical validity of each model's VaR forecasts:

- **Window**: 252-day training window, rolled forward daily
- **VaR levels**: 95% and 99% confidence
- **Procedure**: Refit the model on each window, forecast next-day conditional variance, compute VaR as $\text{VaR}_\alpha = \hat{\sigma}_{t+1} \cdot z_\alpha$ where $z_\alpha$ is the Student-t quantile at the estimated degrees of freedom

Exceedance sequences are evaluated using two formal tests:

**Kupiec POF test** — Kupiec (1995): likelihood-ratio test of whether the observed violation rate matches the nominal confidence level. Tests coverage accuracy.

**Christoffersen independence test** — Christoffersen (1998): likelihood-ratio test of whether violations are independently distributed over time. Clustered violations indicate the model is not capturing volatility regime changes adequately.

Together these tests correspond to Basel III's requirements for internal VaR model validation.

---

## Key Results

| Model | AIC | BIC | Persistence | Kupiec 99% | Christoffersen 99% | Basel III |
|-------|-----|-----|-------------|------------|-------------------|-----------|
| GARCH(1,1) | 11299.83 | 11331.74 | 0.9926 | PASSED | PASSED | GREEN |
| EGARCH(1,1) | 11315.64 | 11347.56 | 0.9742 | REJECT | CLUSTERING | YELLOW |
| GJR-GARCH(1,1) | 11148.03 | 11186.33 | 0.9793 | PASSED | PASSED | GREEN |

**GJR-GARCH(1,1) is the best performing model** across both model selection criteria and backtesting. Its superior fit is attributed to the leverage effect parameter (γ = 0.2585), which captures the asymmetric response of volatility to negative shocks — a dynamic ignored by GARCH(1,1) and imprecisely captured by EGARCH.

---

## Stretch Goal: HAR-RV Benchmark

*Planned extension after core project completion.*

The Heterogeneous Autoregressive model for Realized Volatility (HAR-RV, Corsi 2009) provides a complementary approach to volatility forecasting. Rather than modelling conditional variance parametrically, HAR-RV uses realized volatility computed at daily, weekly, and monthly horizons as predictors in an OLS regression:

$$RV_t = \alpha + \beta_D RV_{t-1} + \beta_W \overline{RV}_{t-5:t-1} + \beta_M \overline{RV}_{t-22:t-1} + \varepsilon_t$$

Forecast accuracy across GARCH variants and HAR-RV will be compared using the **Diebold-Mariano test** (Diebold & Mariano, 1995), which formally tests whether the difference in forecast errors between two models is statistically significant.

---

## Installation

```bash
git clone https://github.com/lukaszchazowiec/GARCH_Volatility_Forecaster.git
cd GARCH_Volatility_Forecaster
pip install -r requirements.txt
```

Then open `notebooks/analysis.ipynb` to run the full narrative analysis.

To rerun the rolling backtest (takes ~8 minutes):
```bash
python src/backtest.py
```

**Requirements**: Python 3.10+, `arch`, `pandas`, `numpy`, `statsmodels`, `matplotlib`, `scipy`, `yfinance`, `tqdm`

---

## Bibliography

### Read

**Engle, R.F. (2001).** GARCH 101: An Introduction to the Use of ARCH/GARCH Models in Applied Econometrics. *Journal of Economic Perspectives*, 15(4), 157–168.
— A practitioner-oriented introduction that grounds GARCH in empirical motivation rather than pure theory. Reading this first clarified the intuition behind volatility clustering and the economic interpretation of GARCH parameters before engaging with the original technical papers.

**Hall, P. & Yao, Q. (2003).** Inference in ARCH and GARCH Models with Heavy-Tailed Errors. *Econometrica*, 71(1), 285–317.
— Examines the behaviour of MLE estimators when the error distribution has heavy tails. Directly relevant given the excess kurtosis in S&P 500 daily returns (kurtosis = 11.70 in this dataset). Informs the choice of Student-t errors over Gaussian in model fitting and the interpretation of QQ plot diagnostics.

---

### Referenced

Original source papers for each model and test implemented in this project. To be read in full during implementation of the corresponding module.

**Engle, R.F. (1982).** Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. *Econometrica*, 50(4), 987–1007.
— Introduces ARCH models and the ARCH-LM test used both to motivate and validate the modelling approach.

**Bollerslev, T. (1986).** Generalized Autoregressive Conditional Heteroscedasticity. *Journal of Econometrics*, 31(3), 307–327.
— Introduces GARCH(p,q). The GARCH(1,1) baseline in this project follows the specification in this paper. Establishes the persistence condition α+β < 1 for covariance stationarity.

**Nelson, D.B. (1991).** Conditional Heteroskedasticity in Asset Returns: A New Approach. *Econometrica*, 59(2), 347–370.
— Introduces EGARCH and the log-variance formulation used to model the leverage effect.

**Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993).** On the Relation Between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779–1801.
— Introduces GJR-GARCH as an alternative leverage-effect specification.

**Kupiec, P.H. (1995).** Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, (2), 73–84.
— Provides the POF likelihood-ratio test for VaR coverage accuracy used in the backtesting module.

**Christoffersen, P.F. (1998).** Evaluating Interval Forecasts. *International Economic Review*, 39(4), 841–862.
— Provides the independence test for VaR violation sequences. Together with Kupiec (1995), satisfies the Basel III joint hypothesis for internal model validation.

**Corsi, F. (2009).** A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
— Introduces HAR-RV, used as a benchmark in the stretch goal.

**Diebold, F.X. & Mariano, R.S. (1995).** Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
— Provides the DM test for formal forecast accuracy comparison across models.

---

## Author

**Łukasz** — second-year student of Economic Analytics (Analityka Gospodarcza) at Uniwersytet Ekonomiczny we Wrocławiu, specialising in financial analysis and risk management. Active member of a student investment fund, risk department.

This project is part of an ongoing effort to build a quantitative finance portfolio targeting risk analyst and junior quant roles.

