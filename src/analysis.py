import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import os


class AdvancedAnalysis:
    """
    Advanced analysis for pairs trading strategy.
    Includes eigenvector evolution, trade statistics, and enhanced metrics.
    """
    
    def __init__(self, data, asset1, asset2, trades_list=None):
        """
        Initialize analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with DatetimeIndex
        asset1 : str
            First asset ticker
        asset2 : str
            Second asset ticker
        trades_list : list
            List of trade dictionaries from backtester
        """
        self.data = data
        self.asset1 = asset1
        self.asset2 = asset2
        self.trades_list = trades_list
        
    def calculate_rolling_eigenvector(self, window=252, step=20):
        """
        Calculate rolling Johansen eigenvector over time.
        
        Parameters:
        -----------
        window : int
            Rolling window size (default 252 = 1 year)
        step : int
            Step size for rolling calculation (default 20 = monthly)
        
        Returns:
        --------
        pd.DataFrame with columns: date, eig1, eig2, beta
        """
        print("\n" + "="*70)
        print("CALCULATING ROLLING EIGENVECTOR (JOHANSEN)")
        print("="*70)
        
        results = []
        
        prices = self.data[[self.asset1, self.asset2]]
        
        for i in range(window, len(prices), step):
            window_data = prices.iloc[i-window:i].values
            
            try:
                # Run Johansen test
                joh_result = coint_johansen(window_data, det_order=0, k_ar_diff=1)
                
                # Get first eigenvector
                eigenvector = joh_result.evec[:, 0]
                
                # Calculate beta (hedge ratio from eigenvector)
                beta = -eigenvector[0] / eigenvector[1] if eigenvector[1] != 0 else np.nan
                
                results.append({
                    'date': prices.index[i],
                    'eig1': eigenvector[0],
                    'eig2': eigenvector[1],
                    'beta': beta,
                    'eigenvalue': joh_result.eig[0]
                })
                
            except Exception as e:
                print(f"Warning: Johansen failed at index {i}: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        print(f"✓ Rolling eigenvector calculated")
        print(f"  Window size: {window} days")
        print(f"  Step size: {step} days")
        print(f"  Total points: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def plot_eigenvector_evolution(self, window=252, step=20, save_path=None):
        """
        Plot evolution of first eigenvector through time.
        """
        eigenvector_df = self.calculate_rolling_eigenvector(window=window, step=step)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        dates = eigenvector_df['date']
        
        # Plot 1: Eigenvector components
        axes[0].plot(dates, eigenvector_df['eig1'], 'b-', 
                    label=f'{self.asset1} component', linewidth=2)
        axes[0].plot(dates, eigenvector_df['eig2'], 'r-', 
                    label=f'{self.asset2} component', linewidth=2)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_title('First Eigenvector Components Evolution (Johansen)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Eigenvector Component', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Implied Beta from eigenvector
        axes[1].plot(dates, eigenvector_df['beta'], 'g-', 
                    label='Beta from Eigenvector', linewidth=2)
        axes[1].set_title('Implied Hedge Ratio from Eigenvector', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Beta (β)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Largest eigenvalue (strength of cointegration)
        axes[2].plot(dates, eigenvector_df['eigenvalue'], 'm-', 
                    label='Largest Eigenvalue', linewidth=2)
        axes[2].set_title('Cointegration Strength (Largest Eigenvalue)', 
                         fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Eigenvalue', fontsize=12)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Eigenvector evolution saved to: {save_path}")
        
        plt.show()
        
        return eigenvector_df
    
    def calculate_trade_statistics(self):
        """
        Calculate detailed trade statistics.
        
        Returns:
        --------
        dict with trade statistics and pd.DataFrame with trade details
        """
        if self.trades_list is None or len(self.trades_list) == 0:
            raise ValueError("No trades available for analysis")
        
        print("\n" + "="*70)
        print("CALCULATING TRADE STATISTICS")
        print("="*70)
        
        trades_df = pd.DataFrame(self.trades_list)
        
        # Identify open and close pairs
        open_trades = trades_df[trades_df['action'] == 'OPEN'].reset_index(drop=True)
        close_trades = trades_df[trades_df['action'] == 'CLOSE'].reset_index(drop=True)
        
        # Calculate P&L for each trade
        trade_pnls = []
        trade_returns = []
        trade_durations = []
        
        for i in range(min(len(open_trades), len(close_trades))):
            open_trade = open_trades.iloc[i]
            close_trade = close_trades.iloc[i]
            
            # P&L calculation
            open_value = (abs(open_trade['asset1_shares'] * open_trade['asset1_price']) + 
                         abs(open_trade['asset2_shares'] * open_trade['asset2_price']))
            close_value = close_trade.get('pnl', 0)
            
            pnl = close_trade['cash'] - open_trade['cash']
            trade_pnls.append(pnl)
            
            # Return calculation
            if open_value > 0:
                trade_return = (pnl / open_value) * 100
                trade_returns.append(trade_return)
            
            # Duration
            duration = (close_trade['date'] - open_trade['date']).days
            trade_durations.append(duration)
        
        # Convert to arrays
        trade_pnls = np.array(trade_pnls)
        trade_returns = np.array(trade_returns)
        trade_durations = np.array(trade_durations)
        
        # Calculate statistics
        winning_trades = trade_pnls[trade_pnls > 0]
        losing_trades = trade_pnls[trade_pnls < 0]
        
        n_trades = len(trade_pnls)
        n_wins = len(winning_trades)
        n_losses = len(losing_trades)
        
        win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0
        
        avg_win = winning_trades.mean() if n_wins > 0 else 0
        avg_loss = losing_trades.mean() if n_losses > 0 else 0
        
        largest_win = winning_trades.max() if n_wins > 0 else 0
        largest_loss = losing_trades.min() if n_losses > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if n_wins > 0 else 0
        gross_loss = abs(losing_trades.sum()) if n_losses > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
        
        # Average duration
        avg_duration = np.mean(trade_durations) if len(trade_durations) > 0 else 0
        
        stats = {
            'Total Trades': n_trades,
            'Winning Trades': n_wins,
            'Losing Trades': n_losses,
            'Win Rate (%)': win_rate,
            'Average Win ($)': avg_win,
            'Average Loss ($)': avg_loss,
            'Largest Win ($)': largest_win,
            'Largest Loss ($)': largest_loss,
            'Gross Profit ($)': gross_profit,
            'Gross Loss ($)': gross_loss,
            'Profit Factor': profit_factor,
            'Average Duration (days)': avg_duration,
            'Total P&L ($)': trade_pnls.sum()
        }
        
        # Create detailed trades dataframe
        detailed_trades = pd.DataFrame({
            'Trade_Number': range(1, n_trades + 1),
            'Open_Date': open_trades['date'].values,
            'Close_Date': close_trades['date'].values,
            'Duration_Days': trade_durations,
            'PnL': trade_pnls,
            'Return_Pct': trade_returns,
            'Signal': open_trades['signal'].values
        })
        
        print("✓ Trade statistics calculated")
        print(f"  Total trades: {n_trades}")
        print(f"  Win rate: {win_rate:.2f}%")
        print(f"  Profit factor: {profit_factor:.3f}")
        print(f"  Average duration: {avg_duration:.1f} days")
        
        return stats, detailed_trades
    
    def plot_trade_distribution(self, save_path=None):
        """
        Plot distribution of trade returns with detailed statistics.
        """
        stats, detailed_trades = self.calculate_trade_statistics()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: P&L Distribution
        axes[0, 0].hist(detailed_trades['PnL'], bins=30, alpha=0.7, 
                       color='blue', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].axvline(x=detailed_trades['PnL'].mean(), color='green', 
                          linestyle='--', linewidth=2, 
                          label=f"Mean: ${detailed_trades['PnL'].mean():,.0f}")
        axes[0, 0].set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('P&L ($)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Return Distribution (%)
        axes[0, 1].hist(detailed_trades['Return_Pct'], bins=30, alpha=0.7, 
                       color='green', edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].axvline(x=detailed_trades['Return_Pct'].mean(), color='blue', 
                          linestyle='--', linewidth=2,
                          label=f"Mean: {detailed_trades['Return_Pct'].mean():.2f}%")
        axes[0, 1].set_title('Trade Return Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Return (%)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cumulative P&L over trades
        cumulative_pnl = detailed_trades['PnL'].cumsum()
        axes[1, 0].plot(detailed_trades['Trade_Number'], cumulative_pnl, 
                       'b-', linewidth=2, marker='o', markersize=4)
        axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].fill_between(detailed_trades['Trade_Number'], 0, cumulative_pnl,
                               where=(cumulative_pnl >= 0), alpha=0.3, color='green')
        axes[1, 0].fill_between(detailed_trades['Trade_Number'], 0, cumulative_pnl,
                               where=(cumulative_pnl < 0), alpha=0.3, color='red')
        axes[1, 0].set_title('Cumulative P&L by Trade', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Trade Number', fontsize=12)
        axes[1, 0].set_ylabel('Cumulative P&L ($)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Trade Duration Distribution
        axes[1, 1].hist(detailed_trades['Duration_Days'], bins=20, alpha=0.7,
                       color='purple', edgecolor='black')
        axes[1, 1].axvline(x=detailed_trades['Duration_Days'].mean(), 
                          color='red', linestyle='--', linewidth=2,
                          label=f"Mean: {detailed_trades['Duration_Days'].mean():.1f} days")
        axes[1, 1].set_title('Trade Duration Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Duration (days)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trade distribution saved to: {save_path}")
        
        plt.show()
        
        return stats, detailed_trades
    
    def print_trade_statistics(self):
        """Print formatted trade statistics."""
        stats, _ = self.calculate_trade_statistics()
        
        print("\n" + "="*70)
        print("DETAILED TRADE STATISTICS")
        print("="*70)
        
        for key, value in stats.items():
            if '$' in key:
                print(f"{key:.<50} ${value:>15,.2f}")
            elif '%' in key or 'Rate' in key:
                print(f"{key:.<50} {value:>15.2f}%")
            elif 'Factor' in key:
                if value == np.inf:
                    print(f"{key:.<50} {'∞':>18}")
                else:
                    print(f"{key:.<50} {value:>18.3f}")
            elif 'days' in key:
                print(f"{key:.<50} {value:>15.1f} days")
            else:
                print(f"{key:.<50} {value:>18,.0f}")
        
        print("="*70)


if __name__ == "__main__":
    print("Advanced Analysis Module")
    print("Provides:")
    print("1. Rolling Eigenvector Evolution (Johansen)")
    print("2. Detailed Trade Statistics")
    print("3. Trade Distribution Analysis")
    print("\nRun from backtester to generate advanced analysis.")