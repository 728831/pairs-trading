# Pairs Trading with Cointegration & Kalman Filters

Statistical arbitrage pairs trading strategy using dual Kalman Filters and cointegration analysis on tech stocks (GOOGL-META).

## 📊 Project Overview

This project implements a market-neutral pairs trading strategy that combines:
- **Johansen Cointegration Test** for pair selection
- **Dual Kalman Filters** for dynamic hedge ratio estimation and signal generation
- **Vector Error Correction Model (VECM)** for spread modeling
- **Sequential Decision Analysis** framework (Powell's methodology)
- **Realistic backtesting** with transaction costs and borrowing fees

## 🎯 Key Results

| Period | Total Return | Sharpe Ratio | Profit Factor | Max Drawdown |
|--------|--------------|--------------|---------------|--------------|
| **Training** | -29.41% | -1.11 | 0.44 | -30.75% |
| **Testing** | -16.12% | -1.05 | 0.02 | -16.12% |
| **Validation** | **+10.78%** | **0.69** | **2.27** | **-6.27%** |

### Selected Pair: GOOGL-META
- **Correlation**: 0.972
- **Cointegrated**: ✅ Both Engle-Granger and Johansen tests
- **Johansen Trace Statistic**: 15.62 (critical value: 15.49)
- **Initial Beta**: 3.7554

## 🏗️ Project Structure

```
pairs-trading-project/
├── data/
│   ├── raw/                      # Downloaded price data (CSV)
│   └── processed/                # Preprocessed data
├── src/
│   ├── data_loader.py           # Data download and preprocessing
│   ├── cointegration.py         # Engle-Granger & Johansen tests
│   ├── kalman_filter.py         # Dual Kalman filter implementation
│   ├── strategy.py              # Trading strategy logic
│   ├── backtester.py            # Backtesting engine with costs
│   ├── analysis.py              # Advanced analysis (eigenvector, trades)
│   └── run_full_backtest.py     # Main execution script
├── results/
│   ├── figures/                 # All plots and visualizations
│   └── tables/                  # CSV results and metrics
├── notebooks/
│   └── pairs_trading_report.ipynb  # Complete analysis report
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/728831/pairs-trading.git
cd pairs-trading
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
yfinance>=0.2.30
```

## 📈 Usage

### Run Complete Backtest

```bash
cd src
python run_full_backtest.py
```

This will:
1. Download 15 years of daily price data for AAPL, MSFT, GOOGL, META
2. Split data: 60% Train / 20% Test / 20% Validation
3. Run cointegration analysis (Engle-Granger + Johansen)
4. Execute strategy with dual Kalman filters
5. Generate all performance metrics and plots
6. Save results to `results/` directory

### View Report

Open the Jupyter notebook for detailed analysis:
```bash
jupyter notebook notebooks/pairs_trading_report.ipynb
```

### Run Individual Modules

**Download data only:**
```python
from data_loader import DataLoader

loader = DataLoader(['GOOGL', 'META'])
prices = loader.download_data(save=True)
train, test, val = loader.split_data()
```

**Test cointegration:**
```python
from cointegration import CointegrationAnalysis

analysis = CointegrationAnalysis(train)
results = analysis.test_all_pairs(corr_threshold=0.7)
best_pair = analysis.get_best_pairs(top_n=1)
```

**Run strategy:**
```python
from strategy import PairsTradingStrategy

strategy = PairsTradingStrategy(
    asset1='GOOGL',
    asset2='META',
    data=train,
    entry_threshold=2.0,
    exit_threshold=0.5
)
results = strategy.run_strategy()
```

## 🔬 Methodology

### 1. Pair Selection
- **Correlation screening**: Filter pairs with correlation > 0.7
- **Engle-Granger test**: ADF test on OLS residuals (p-value < 0.05)
- **Johansen test**: Trace statistic exceeds critical value
- **Selection**: GOOGL-META (only pair passing both tests)

### 2. Kalman Filter #1: Dynamic Hedge Ratio

State space model:
```
Observation: y_t = β_t * x_t + v_t,  v_t ~ N(0, Ve)
State:       β_t = β_{t-1} + w_t,    w_t ~ N(0, δ)
```

Parameters:
- Process noise (δ): 2×10⁻⁵
- Measurement noise (Ve): 5×10⁻⁵
- Initial β: 3.7554 (from Johansen eigenvector)

### 3. VECM (Vector Error Correction Model)

Models short-run dynamics and long-run equilibrium:
```
ΔY_t = α * β' * Y_{t-1} + Γ * ΔY_{t-1} + ε_t
```

Results:
- Error correction speed (α): 0.0119 (training)
- Half-life of mean reversion: ~58 days

### 4. Kalman Filter #2: Trading Signals

Filters the spread and generates z-scores:
```
Z_t = (μ_t - mean(μ)) / std(μ)
```

Trading rules:
- **Long spread**: Z < -2.0σ (spread undervalued)
- **Short spread**: Z > +2.0σ (spread overvalued)
- **Exit**: |Z| < 0.5σ (near equilibrium)

### 5. Sequential Decision Analysis

Three interconnected models:
- **Model A**: Hedge ratio estimation (state: β_t, P_t)
- **Model B**: Signal generation (state: μ_t, P_t^(s), position)
- **Model C**: Complete system (integration + execution)

Each model follows the 6-step process: Initialize → Predict → Observe → Innovate → Update → Decide

## 📊 Results Summary

### Performance Metrics

**Validation Period (Profitable)**:
- Total Return: +10.78%
- Annualized: +3.87%
- Sharpe Ratio: 0.69
- Win Rate: 51.6%
- Profit Factor: 2.27
- Average Win: $15,644
- Average Loss: $7,346

### Key Findings

✅ **What Worked**:
- Dynamic hedge ratios outperform static OLS
- Kalman filtering reduces spread noise by 40%
- Strategy profitable in validation period
- Risk-adjusted returns acceptable (Sharpe = 0.69)

❌ **Challenges**:
- High transaction costs (15.6% of capital in training)
- Slow mean reversion (half-life ~58 days)
- Regime-dependent performance
- Unstable cointegration during market stress

### Transaction Cost Breakdown

| Period | Total Commissions | % of Capital | Trades |
|--------|------------------|--------------|--------|
| Training | $156,398 | 15.6% | 96 |
| Testing | $26,194 | 2.6% | 14 |
| Validation | $64,737 | 6.5% | 31 |

## 🎨 Visualizations

The project generates 14+ plots including:

1. **Correlation Matrix** - Heatmap of asset correlations
2. **Eigenvector Evolution** - Rolling Johansen eigenvector components
3. **Strategy Overview** - Prices, hedge ratio, spread, z-scores, signals
4. **Backtest Performance** - Portfolio value, drawdown, returns distribution
5. **Trade Distribution** - P&L histogram, cumulative P&L, trade duration
6. **Combined Performance** - All periods on single chart

All figures saved to `results/figures/`

## 📝 Report

Complete 18-page analysis report available in:
- **Jupyter Notebook**: `notebooks/pairs_trading_report.ipynb`
- **Contains**: 
  - Executive summary
  - Pair selection methodology
  - Sequential Decision Analysis (3 models)
  - Kalman filter mathematical formulation
  - VECM implementation
  - Results by period
  - Critical analysis and insights
  - Conclusions and recommendations

## ⚙️ Configuration

### Kalman Filter Parameters

```python
# Hedge Ratio Filter
delta_hedge = 2e-5      # Process noise
Ve_hedge = 5e-5         # Measurement noise

# Signal Filter  
delta_signal = 5e-5     # Process noise
Ve_signal = 1e-5        # Measurement noise
```

### Trading Parameters

```python
entry_threshold = 2.0   # Entry at ±2σ
exit_threshold = 0.5    # Exit at ±0.5σ
position_pct = 0.80     # Use 80% of capital
commission_rate = 0.00125  # 0.125% per trade
borrow_rate = 0.0025    # 0.25% annualized
```

## 🔧 Future Enhancements

1. **Regime Detection**: Implement HMM to detect regime changes
2. **Adaptive Thresholds**: Dynamic entry/exit based on volatility
3. **Stop-Loss**: Add maximum loss per trade (5%)
4. **Multi-Pair**: Trade multiple cointegrated pairs simultaneously
5. **Cost Optimization**: Test different holding periods (5-10 days)
6. **Non-Linear Filters**: Extended Kalman Filter or Particle Filter

## 📚 References

### Academic Papers
- Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction"
- Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors"
- Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems"

### Books
- Powell, W.B. (2011). "Approximate Dynamic Programming"
- Gatev, E., Goetzmann, W.N., and Rouwenhorst, K.G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"

### Libraries
- [Statsmodels](https://www.statsmodels.org/) - Econometric models
- [yfinance](https://github.com/ranaroussi/yfinance) - Market data
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data analysis
