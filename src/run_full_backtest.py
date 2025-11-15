import sys
sys.path.append('src')

from data_loader import DataLoader
from strategy import PairsTradingStrategy
from backtester import PairsBacktester
from analysis import AdvancedAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def run_complete_backtest_v2():
    
    print("="*70)
    print("COMPLETE PAIRS TRADING BACKTEST V2 (ENHANCED)")
    print("="*70)
    
    # 1. LOAD DATA
    print("\n[1/7] Loading data...")
    loader = DataLoader(['AAPL', 'MSFT', 'GOOGL', 'META'])
    
    # Force download fresh data
    prices = loader.download_data(save=True)
    
    # Split data
    train, test, val = loader.split_data()
    
    # 1.5. RUN COINTEGRATION ANALYSIS ON TRAINING DATA
    print("\n[1.5/7] Running cointegration analysis...")
    print("-"*70)
    
    from cointegration import CointegrationAnalysis
    
    coint_analysis = CointegrationAnalysis(train)
    coint_analysis.calculate_correlations()
    coint_analysis.plot_correlation_matrix(save_path='results/figures/correlation_matrix.png')
    
    # Test all pairs and get the best one
    all_results = coint_analysis.test_all_pairs(corr_threshold=0.7, significance=0.05)
    coint_analysis.save_results('results/tables/cointegration_results.csv')
    
    best_pairs = coint_analysis.get_best_pairs(top_n=1, method='both')
    
    if len(best_pairs) == 0:
        print("\n⚠ WARNING: No cointegrated pairs found!")
        print("Using highest correlation pair instead...")
        best_pairs = all_results.sort_values('Correlation', ascending=False).head(1)
    
    # Extract best pair
    best_pair = best_pairs.iloc[0]
    asset1 = best_pair['Asset1']
    asset2 = best_pair['Asset2']
    
    print(f"\n✓ Selected Pair: {asset1} - {asset2}")
    print(f"  Correlation: {best_pair['Correlation']:.4f}")
    print(f"  EG Cointegrated: {best_pair['EG_Cointegrated']}")
    print(f"  Johansen Cointegrated: {best_pair['JOH_Cointegrated']}")
    
    # Get Johansen result for this pair
    joh_result = coint_analysis.johansen_test(asset1, asset2, significance=0.05)
    
    print(f"\n✓ Johansen Test Results:")
    print(f"  Cointegrated: {joh_result['is_cointegrated']}")
    print(f"  Eigenvector: [{joh_result['eigenvectors'][0]:.4f}, {joh_result['eigenvectors'][1]:.4f}]")
    if joh_result['eigenvectors'][1] != 0:
        print(f"  Initial beta: {-joh_result['eigenvectors'][0]/joh_result['eigenvectors'][1]:.4f}")
    
    # 2. RUN STRATEGY ON TRAIN SET
    print("\n[2/7] Running strategy on TRAINING set...")
    print("-"*70)
    
    # THETA PARAMETERS (Kalman Filter noise parameters)
    DELTA_HEDGE = 2e-5      
    VE_HEDGE = 5e-5         
    DELTA_SIGNAL = 5e-5    
    VE_SIGNAL = 1e-5        
    
    print(f"Using Theta Parameters:")
    print(f"  δ_hedge  (delta_hedge):  {DELTA_HEDGE:.0e}")
    print(f"  Ve_hedge (Ve_hedge):     {VE_HEDGE:.0e}")
    print(f"  δ_signal (delta_signal): {DELTA_SIGNAL:.0e}")
    print(f"  Ve_signal (Ve_signal):   {VE_SIGNAL:.0e}")
    
    strategy_train = PairsTradingStrategy(
        asset1=asset1,
        asset2=asset2,
        data=train,
        entry_threshold=2.0,
        exit_threshold=0.5,
        delta_hedge=DELTA_HEDGE,
        Ve_hedge=VE_HEDGE,
        delta_signal=DELTA_SIGNAL,
        Ve_signal=VE_SIGNAL,
        johansen_result=joh_result if joh_result['is_cointegrated'] else None
    )
    
    results_train = strategy_train.run_strategy()
    
    # 3. BACKTEST ON TRAIN
    print("\n[3/7] Backtesting TRAINING period...")
    print("-"*70)
    
    backtester_train = PairsBacktester(
        data=train,
        asset1=asset1,
        asset2=asset2,
        signals=strategy_train.signals,
        hedge_ratios=strategy_train.hedge_ratios,
        initial_capital=1_000_000,
        position_pct=0.8,
        commission_rate=0.00125,
        borrow_rate=0.0025
    )
    
    results_train_bt = backtester_train.run_backtest()
    metrics_train = backtester_train.calculate_metrics()
    
    print("\n📊 TRAINING SET RESULTS:")
    backtester_train.print_metrics()
    
    # 3.5. ADVANCED ANALYSIS ON TRAIN
    print("\n[3.5/7] Running advanced analysis on TRAINING set...")
    print("-"*70)
    
    analysis_train = AdvancedAnalysis(
        data=train,
        asset1=asset1,
        asset2=asset2,
        trades_list=backtester_train.trades
    )
    
    # Eigenvector evolution
    eigenvector_train = analysis_train.plot_eigenvector_evolution(
        window=252,
        step=20,
        save_path='results/figures/eigenvector_evolution_train.png'
    )
    eigenvector_train.to_csv('results/tables/eigenvector_evolution_train.csv', index=False)
    
    # Trade statistics
    trade_stats_train, detailed_trades_train = analysis_train.plot_trade_distribution(
        save_path='results/figures/trade_distribution_train.png'
    )
    analysis_train.print_trade_statistics()
    detailed_trades_train.to_csv('results/tables/detailed_trades_train.csv', index=False)
    
    # 4. RUN ON TEST SET
    print("\n[4/7] Running strategy on TEST set...")
    print("-"*70)
    
    strategy_test = PairsTradingStrategy(
        asset1=asset1,
        asset2=asset2,
        data=test,
        entry_threshold=2.5,
        exit_threshold=1.5,
        delta_hedge=DELTA_HEDGE,
        Ve_hedge=VE_HEDGE,
        delta_signal=DELTA_SIGNAL,
        Ve_signal=VE_SIGNAL,
        johansen_result=joh_result if joh_result['is_cointegrated'] else None
    )
    
    results_test = strategy_test.run_strategy()
    
    backtester_test = PairsBacktester(
        data=test,
        asset1=asset1,
        asset2=asset2,
        signals=strategy_test.signals,
        hedge_ratios=strategy_test.hedge_ratios,
        initial_capital=1_000_000,
        position_pct=0.8,
        commission_rate=0.00125,
        borrow_rate=0.0025
    )
    
    results_test_bt = backtester_test.run_backtest()
    metrics_test = backtester_test.calculate_metrics()
    
    print("\n📊 TEST SET RESULTS:")
    backtester_test.print_metrics()
    
    # Advanced analysis on test
    analysis_test = AdvancedAnalysis(
        data=test,
        asset1=asset1,
        asset2=asset2,
        trades_list=backtester_test.trades
    )
    
    eigenvector_test = analysis_test.plot_eigenvector_evolution(
        window=252,
        step=20,
        save_path='results/figures/eigenvector_evolution_test.png'
    )
    eigenvector_test.to_csv('results/tables/eigenvector_evolution_test.csv', index=False)
    
    trade_stats_test, detailed_trades_test = analysis_test.plot_trade_distribution(
        save_path='results/figures/trade_distribution_test.png'
    )
    analysis_test.print_trade_statistics()
    detailed_trades_test.to_csv('results/tables/detailed_trades_test.csv', index=False)
    
    # 5. RUN ON VALIDATION SET
    print("\n[5/7] Running strategy on VALIDATION set...")
    print("-"*70)
    
    strategy_val = PairsTradingStrategy(
        asset1=asset1,
        asset2=asset2,
        data=val,
        entry_threshold=2.0,
        exit_threshold=0.5,
        delta_hedge=DELTA_HEDGE,
        Ve_hedge=VE_HEDGE,
        delta_signal=DELTA_SIGNAL,
        Ve_signal=VE_SIGNAL,
        johansen_result=joh_result if joh_result['is_cointegrated'] else None
    )
    
    results_val = strategy_val.run_strategy()
    
    backtester_val = PairsBacktester(
        data=val,
        asset1=asset1,
        asset2=asset2,
        signals=strategy_val.signals,
        hedge_ratios=strategy_val.hedge_ratios,
        initial_capital=1_000_000,
        position_pct=0.8,
        commission_rate=0.00125,
        borrow_rate=0.0025
    )
    
    results_val_bt = backtester_val.run_backtest()
    metrics_val = backtester_val.calculate_metrics()
    
    print("\n📊 VALIDATION SET RESULTS:")
    backtester_val.print_metrics()
    
    # Advanced analysis on validation
    analysis_val = AdvancedAnalysis(
        data=val,
        asset1=asset1,
        asset2=asset2,
        trades_list=backtester_val.trades
    )
    
    eigenvector_val = analysis_val.plot_eigenvector_evolution(
        window=252,
        step=20,
        save_path='results/figures/eigenvector_evolution_val.png'
    )
    eigenvector_val.to_csv('results/tables/eigenvector_evolution_val.csv', index=False)
    
    trade_stats_val, detailed_trades_val = analysis_val.plot_trade_distribution(
        save_path='results/figures/trade_distribution_val.png'
    )
    analysis_val.print_trade_statistics()
    detailed_trades_val.to_csv('results/tables/detailed_trades_val.csv', index=False)
    
    # 6. COMPARISON TABLES
    print("\n[6/7] Creating comparison tables...")
    print("-"*70)
    
    # Performance comparison
    performance_comparison = pd.DataFrame({
        'Train': metrics_train,
        'Test': metrics_test,
        'Validation': metrics_val
    }).T
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(performance_comparison.to_string())
    performance_comparison.to_csv('results/tables/performance_comparison.csv')
    
    # Trade statistics comparison
    trade_stats_comparison = pd.DataFrame({
        'Train': trade_stats_train,
        'Test': trade_stats_test,
        'Validation': trade_stats_val
    }).T
    
    print("\n" + "="*70)
    print("TRADE STATISTICS COMPARISON")
    print("="*70)
    print(trade_stats_comparison.to_string())
    trade_stats_comparison.to_csv('results/tables/trade_stats_comparison.csv')
    
    # 7. GENERATE ALL PLOTS
    print("\n[7/7] Generating all plots...")
    print("-"*70)
    
    # Strategy and backtest plots
    backtester_train.plot_results(save_path='results/figures/backtest_train.png')
    strategy_train.plot_strategy_overview(save_path='results/figures/strategy_train.png')
    
    backtester_test.plot_results(save_path='results/figures/backtest_test.png')
    strategy_test.plot_strategy_overview(save_path='results/figures/strategy_test.png')
    
    backtester_val.plot_results(save_path='results/figures/backtest_validation.png')
    strategy_val.plot_strategy_overview(save_path='results/figures/strategy_validation.png')
    
    # Combined performance plot
    plot_combined_performance(backtester_train, backtester_test, backtester_val, train, test, val)
    
    print("\n" + "="*70)
    print("✅ COMPLETE BACKTEST V2 FINISHED!")
    print("="*70)
    print("\nAll results saved to:")
    print("  - results/figures/")
    print("  - results/tables/")
    print("\nNew files generated:")
    print("  ✓ eigenvector_evolution_[train/test/val].png")
    print("  ✓ trade_distribution_[train/test/val].png")
    print("  ✓ detailed_trades_[train/test/val].csv")
    print("  ✓ trade_stats_comparison.csv")
    
    return {
        'train': {
            'strategy': strategy_train, 
            'backtest': backtester_train,
            'analysis': analysis_train,
            'trade_stats': trade_stats_train
        },
        'test': {
            'strategy': strategy_test, 
            'backtest': backtester_test,
            'analysis': analysis_test,
            'trade_stats': trade_stats_test
        },
        'validation': {
            'strategy': strategy_val, 
            'backtest': backtester_val,
            'analysis': analysis_val,
            'trade_stats': trade_stats_val
        }
    }


def plot_combined_performance(bt_train, bt_test, bt_val, train, test, val):
    """Plot combined performance across all periods."""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Combine portfolio values
    pv_train = pd.Series(bt_train.portfolio_value, index=train.index[-len(bt_train.portfolio_value):])
    pv_test = pd.Series(bt_test.portfolio_value, index=test.index[-len(bt_test.portfolio_value):])
    pv_val = pd.Series(bt_val.portfolio_value, index=val.index[-len(bt_val.portfolio_value):])
    
    # Scale test and val to continue from train
    pv_test_scaled = pv_test * (pv_train.iloc[-1] / pv_test.iloc[0])
    pv_val_scaled = pv_val * (pv_test_scaled.iloc[-1] / pv_val.iloc[0])
    
    # Plot 1: Portfolio value across all periods
    axes[0].plot(pv_train.index, pv_train.values, 'b-', linewidth=2, label='Train')
    axes[0].plot(pv_test_scaled.index, pv_test_scaled.values, 'g-', linewidth=2, label='Test')
    axes[0].plot(pv_val_scaled.index, pv_val_scaled.values, 'r-', linewidth=2, label='Validation')
    
    # Add vertical lines for period separations
    axes[0].axvline(x=train.index[-1], color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=test.index[-1], color='gray', linestyle='--', alpha=0.5)
    
    axes[0].set_title('Portfolio Value - All Periods (Scaled)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
    
    # Plot 2: Returns comparison
    returns_train = pd.Series(bt_train.returns, index=train.index[-len(bt_train.returns):]) * 100
    returns_test = pd.Series(bt_test.returns, index=test.index[-len(bt_test.returns):]) * 100
    returns_val = pd.Series(bt_val.returns, index=val.index[-len(bt_val.returns):]) * 100
    
    axes[1].hist(returns_train.values, bins=30, alpha=0.5, label='Train', color='blue')
    axes[1].hist(returns_test.values, bins=30, alpha=0.5, label='Test', color='green')
    axes[1].hist(returns_val.values, bins=30, alpha=0.5, label='Validation', color='red')
    
    axes[1].set_title('Daily Returns Distribution by Period', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Daily Return (%)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = 'results/figures/combined_performance.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined performance plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    results = run_complete_backtest_v2()