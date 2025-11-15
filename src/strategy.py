
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM
import os
import warnings
warnings.filterwarnings('ignore')

from kalman_filter import KalmanFilterHedgeRatio, KalmanFilterTradingSignals, calculate_spread_vecm


class PairsTradingStrategy:

    def __init__(self, asset1, asset2, data, 
                 delta_hedge=2e-5, Ve_hedge=5e-5,
                 delta_signal=6e-5, Ve_signal=1e-5,
                 entry_threshold=2.0, exit_threshold=0.5,
                 stop_loss=None, take_profit=None,
                 johansen_result=None):

        self.asset1 = asset1
        self.asset2 = asset2
        self.data = data
        self.johansen_result = johansen_result
        
        # Kalman Filters
        self.kf_hedge = KalmanFilterHedgeRatio(delta=delta_hedge, Ve=Ve_hedge)
        self.kf_signal = KalmanFilterTradingSignals(delta=delta_signal, Ve=Ve_signal)
        
        # Strategy parameters
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Results
        self.hedge_ratios = None
        self.spread = None
        self.filtered_spread = None
        self.z_scores = None
        self.signals = None
        self.vecm_model = None
        self.vecm_result = None
        self.alpha_coefficients = None  # Error correction speeds
        
    def calculate_dynamic_hedge_ratio(self):

        print("\n" + "="*70)
        print("STEP 1: ESTIMATING DYNAMIC HEDGE RATIO (Kalman Filter #1)")
        print("="*70)
        
        prices1 = self.data[self.asset1].values
        prices2 = self.data[self.asset2].values
        
        # Get initial hedge ratio from Johansen if available
        initial_beta = None
        if self.johansen_result is not None:
            # Johansen eigenvector: [w1, w2] where spread = w1*P1 + w2*P2
            # Convert to hedge ratio: beta = -w1/w2 (so spread = P1 - beta*P2)
            eigenvector = self.johansen_result['eigenvectors']
            initial_beta = -eigenvector[0] / eigenvector[1]
            print(f"✓ Using Johansen eigenvector for initialization")
            print(f"  Eigenvector: [{eigenvector[0]:.4f}, {eigenvector[1]:.4f}]")
            print(f"  Initial β from Johansen: {initial_beta:.4f}")
        else:
            print(f"⚠ No Johansen result provided, using OLS initialization")
        
        # Run Kalman Filter
        hedge_ratios, P_est, K_gains = self.kf_hedge.filter(prices1, prices2, initial_beta=initial_beta)
        
        self.hedge_ratios = hedge_ratios
        
        print(f"✓ Hedge ratio estimated")
        print(f"  Mean β: {np.mean(hedge_ratios):.4f}")
        print(f"  Std β: {np.std(hedge_ratios):.4f}")
        print(f"  Min β: {np.min(hedge_ratios):.4f}")
        print(f"  Max β: {np.max(hedge_ratios):.4f}")
        
        return hedge_ratios
    
    def fit_vecm(self, data_for_vecm=None):

        print("\n" + "="*70)
        print("STEP 2: FITTING VECM MODEL")
        print("="*70)
        
        if data_for_vecm is None:
            data_for_vecm = self.data[[self.asset1, self.asset2]]
        
        # Fit VECM
        try:
            self.vecm_model = VECM(data_for_vecm, k_ar_diff=1, coint_rank=1, deterministic='ci')
            self.vecm_result = self.vecm_model.fit()
            
            print(f"✓ VECM model fitted")
            print(f"  Cointegration rank: 1")
            print(f"  Lag order: 1")
            
            # Extract error correction coefficients (alpha)
            # Alpha represents the speed of mean reversion
            # Larger |alpha| = faster mean reversion
            self.alpha_coefficients = self.vecm_result.alpha
            
            print(f"  Error correction coefficients (α):")
            print(f"    {self.asset1}: {self.alpha_coefficients[0, 0]:.4f}")
            print(f"    {self.asset2}: {self.alpha_coefficients[1, 0]:.4f}")
            print(f"  Mean reversion speed: {np.mean(np.abs(self.alpha_coefficients)):.4f}")
            
            # Store beta from VECM for comparison
            vecm_beta = self.vecm_result.beta[0, 0]
            print(f"  VECM cointegrating vector (β): {vecm_beta:.4f}")
            
            return self.vecm_result
            
        except Exception as e:
            print(f"⚠ VECM fitting failed: {e}")
            print("  Using simple spread calculation instead")
            self.alpha_coefficients = None
            return None
    
    def calculate_spread(self):

        print("\n" + "="*70)
        print("STEP 3: CALCULATING SPREAD")
        print("="*70)
        
        if self.hedge_ratios is None:
            self.calculate_dynamic_hedge_ratio()
        
        prices1 = self.data[self.asset1].values
        prices2 = self.data[self.asset2].values
        
        # Adjust for length difference (Kalman Filter has one extra value at start)
        min_len = min(len(prices1), len(prices2), len(self.hedge_ratios))
        prices1 = prices1[:min_len]
        prices2 = prices2[:min_len]
        hedge_ratios_adj = self.hedge_ratios[:min_len]
        
        # Calculate spread
        self.spread = calculate_spread_vecm(prices1, prices2, hedge_ratios_adj)
        
        print(f"✓ Spread calculated")
        print(f"  Length: {len(self.spread)}")
        print(f"  Mean: {np.mean(self.spread):.4f}")
        print(f"  Std: {np.std(self.spread):.4f}")
        
        return self.spread
    
    def filter_spread_signals(self):

        print("\n" + "="*70)
        print("STEP 4: FILTERING SPREAD (Kalman Filter #2)")
        print("="*70)
        
        if self.spread is None:
            self.calculate_spread()
        
        # Apply Kalman Filter
        self.filtered_spread = self.kf_signal.filter(self.spread)
        
        print(f"✓ Spread filtered")
        print(f"  Noise reduction: {(1 - np.std(self.filtered_spread)/np.std(self.spread))*100:.2f}%")
        
        return self.filtered_spread
    
    def calculate_z_scores(self, window=20):

        print("\n" + "="*70)
        print("STEP 5: CALCULATING Z-SCORES")
        print("="*70)
        
        if self.filtered_spread is None:
            self.filter_spread_signals()
        
        # Calculate rolling statistics
        spread_series = pd.Series(self.filtered_spread)
        rolling_mean = spread_series.rolling(window=window).mean()
        rolling_std = spread_series.rolling(window=window).std()
        
        # Calculate z-scores
        self.z_scores = (spread_series - rolling_mean) / rolling_std
        self.z_scores = self.z_scores.fillna(0).values
        
        print(f"✓ Z-scores calculated")
        print(f"  Window: {window} days")
        print(f"  Mean Z: {np.mean(self.z_scores):.4f}")
        print(f"  Std Z: {np.std(self.z_scores):.4f}")
        print(f"  Max |Z|: {np.max(np.abs(self.z_scores)):.4f}")
        
        return self.z_scores
    
    def generate_signals(self):

        print("\n" + "="*70)
        print("STEP 6: GENERATING TRADING SIGNALS")
        print("="*70)
        
        if self.z_scores is None:
            self.calculate_z_scores()
        
        signals = np.zeros(len(self.z_scores))
        position = 0  # Current position
        
        for i in range(len(self.z_scores)):
            z = self.z_scores[i]
            
            if position == 0:  # No position
                if z < -self.entry_threshold:
                    position = 1  # Long spread
                elif z > self.entry_threshold:
                    position = -1  # Short spread
                    
            elif position == 1:  # Long spread
                if abs(z) < self.exit_threshold or z > self.entry_threshold:
                    position = 0  # Exit
                    
            elif position == -1:  # Short spread
                if abs(z) < self.exit_threshold or z < -self.entry_threshold:
                    position = 0  # Exit
            
            signals[i] = position
        
        self.signals = signals
        
        # Calculate signal statistics
        n_long = np.sum(signals == 1)
        n_short = np.sum(signals == -1)
        n_flat = np.sum(signals == 0)
        
        print(f"✓ Signals generated")
        print(f"  Entry threshold: ±{self.entry_threshold}")
        print(f"  Exit threshold: ±{self.exit_threshold}")
        print(f"  Long positions: {n_long} days ({n_long/len(signals)*100:.1f}%)")
        print(f"  Short positions: {n_short} days ({n_short/len(signals)*100:.1f}%)")
        print(f"  Flat: {n_flat} days ({n_flat/len(signals)*100:.1f}%)")
        
        return signals
    
    def run_strategy(self):

        print("\n" + "="*70)
        print(f"RUNNING PAIRS TRADING STRATEGY: {self.asset1} - {self.asset2}")
        print("="*70)
        
        # Run all steps
        self.calculate_dynamic_hedge_ratio()
        self.fit_vecm()
        self.calculate_spread()
        self.filter_spread_signals()
        self.calculate_z_scores()
        self.generate_signals()
        
        print("\n" + "="*70)
        print("✅ STRATEGY EXECUTION COMPLETE")
        print("="*70)
        
        return {
            'hedge_ratios': self.hedge_ratios,
            'spread': self.spread,
            'filtered_spread': self.filtered_spread,
            'z_scores': self.z_scores,
            'signals': self.signals
        }
    
    def plot_strategy_overview(self, save_path=None):

        if self.signals is None:
            raise ValueError("Run strategy first!")
        
        # Get minimum length across all arrays
        n = min(
            len(self.signals),
            len(self.hedge_ratios),
            len(self.spread),
            len(self.filtered_spread),
            len(self.z_scores),
            len(self.data)
        )
        
        # Truncate all arrays to same length
        dates = self.data.index[-n:]
        prices1 = self.data[self.asset1].values[-n:]
        prices2 = self.data[self.asset2].values[-n:]
        hedge_ratios = self.hedge_ratios[-n:]
        spread = self.spread[-n:]
        filtered_spread = self.filtered_spread[-n:]
        z_scores = self.z_scores[-n:]
        signals = self.signals[-n:]
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 14))
        
        # Plot 1: Prices
        axes[0].plot(dates, prices1, label=self.asset1, linewidth=2)
        axes[0].plot(dates, prices2, label=self.asset2, linewidth=2)
        axes[0].set_title('Asset Prices', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Hedge Ratio
        axes[1].plot(dates, hedge_ratios, 'b-', linewidth=2)
        axes[1].set_title('Dynamic Hedge Ratio (Kalman Filter)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Beta (β)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Spread and Filtered Spread
        axes[2].plot(dates, spread, 'gray', alpha=0.5, label='Raw Spread', linewidth=1)
        axes[2].plot(dates, filtered_spread, 'b-', label='Filtered Spread', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_title('Spread Evolution', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Spread', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Z-scores and Signals
        axes[3].plot(dates, z_scores, 'k-', linewidth=1.5, label='Z-score')
        axes[3].axhline(y=self.entry_threshold, color='r', linestyle='--', label='Entry Threshold')
        axes[3].axhline(y=-self.entry_threshold, color='r', linestyle='--')
        axes[3].axhline(y=self.exit_threshold, color='g', linestyle='--', label='Exit Threshold')
        axes[3].axhline(y=-self.exit_threshold, color='g', linestyle='--')
        axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Highlight positions
        long_mask = signals == 1
        short_mask = signals == -1
        axes[3].fill_between(dates, -5, 5, where=long_mask, alpha=0.2, color='green', label='Long Spread')
        axes[3].fill_between(dates, -5, 5, where=short_mask, alpha=0.2, color='red', label='Short Spread')
        
        axes[3].set_title('Z-scores and Trading Signals', fontsize=14, fontweight='bold')
        axes[3].set_ylabel('Z-score', fontsize=12)
        axes[3].set_xlabel('Date', fontsize=12)
        axes[3].set_ylim([-5, 5])
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Strategy overview saved to: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    from data_loader import DataLoader
    import glob
    
    print("="*70)
    print("PAIRS TRADING STRATEGY EXECUTION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    loader = DataLoader(['AAPL', 'MSFT', 'GOOGL', 'META'])
    
    csv_files = glob.glob('data/raw/*.csv')
    if csv_files:
        prices = loader.load_data(csv_files[0])
    else:
        prices = loader.download_data()
    
    # Use training data
    train, test, val = loader.split_data()
    
    # Initialize strategy (using best pair: 
    strategy = PairsTradingStrategy(
        asset1='GOOGL',
        asset2='META',
        data=train,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    # Run strategy
    results = strategy.run_strategy()
    
    # Plot results
    strategy.plot_strategy_overview(save_path='results/figures/strategy_overview_train.png')
    
    print("\n✅ Strategy execution complete!")
    print("Next step: Run backtesting with transaction costs")