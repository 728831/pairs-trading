Project Structure
pairs-trading-project/
│
├── data/                          
│   ├── raw/                       
│   └── processed/                 
│
├── notebooks/                     
│   ├── 01_data_exploration.ipynb
│   ├── 02_pair_selection.ipynb
│   └── 03_strategy_analysis.ipynb
│
├── src/                          
│   ├── __init__.py
│   ├── data_loader.py            
│   ├── cointegration.py          
│   ├── kalman_filter.py          
│   ├── strategy.py               
│   ├── backtester.py             
│   └── performance.py            
│
├── results/                       
│   ├── figures/
│   └── tables/
│
├── report/                      
│   └── executive_report.pdf
│
├── requirements.txt              
├── .gitignore                    
└── README.md                     

Data Requirements

15 years of daily price data
Technology sector stocks
Data split: 60% Train, 20% Test, 20% Validation

Strategy Parameters

Commission: 0.125% per trade
Borrow Rate: 0.25% annualized
Initial Capital: $1,000,000
Position Size: 80% of capital (40% per asset)

Key Features

Johansen Cointegration Test
Engle-Granger Cointegration Test
Dynamic Hedge Ratio via Kalman Filter
VECM-based Trading Signals
Walk-forward Analysis
Realistic Transaction Costs

Performance Metrics

Sharpe Ratio
Sortino Ratio
Calmar Ratio
Maximum Drawdown
Win Rate
Profit Factor