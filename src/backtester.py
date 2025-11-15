
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class PairsBacktester:
    
    def __init__(self, data, asset1, asset2, signals, hedge_ratios,
                 initial_capital=1_000_000, position_pct=0.8,
                 commission_rate=0.00125, borrow_rate=0.0025):

        self.data = data
        self.asset1 = asset1
        self.asset2 = asset2
        
        # Align signals and hedge ratios with data
        n = min(len(signals), len(hedge_ratios), len(data))
        self.signals = signals[-n:]
        self.hedge_ratios = hedge_ratios[-n:]
        self.data = data.iloc[-n:]
        
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.commission_rate = commission_rate
        self.borrow_rate = borrow_rate / 252  # Daily borrow rate
        
        # Results
        self.portfolio_value = None
        self.returns = None
        self.positions = None
        self.trades = []
        self.metrics = {}
        
    def run_backtest(self):

        print("\n" + "="*70)
        print("BACKTESTING WITH TRANSACTION COSTS")
        print("="*70)
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Position Size: {self.position_pct*100:.0f}% of capital")
        print(f"Commission Rate: {self.commission_rate*100:.3f}%")
        print(f"Borrow Rate: {self.borrow_rate*252*100:.3f}% annualized")
        
        # Initialize
        n = len(self.data)
        portfolio_values = np.zeros(n)
        cash = self.initial_capital
        
        # Position tracking
        position_asset1 = 0  # Number of shares
        position_asset2 = 0
        current_signal = 0
        
        #beta_eff = float(abs(beta)) if beta != 0 else 1.0
        #s = capital_to_use / (abs(prices1[t]) + beta_eff * abs(prices2[t]))
        # Get prices
        prices1 = self.data[self.asset1].values
        prices2 = self.data[self.asset2].values
        
        # Simulation
        for t in range(n):
            signal = self.signals[t]
            beta = self.hedge_ratios[t]
            
            # Check if signal changed
            if signal != current_signal:
                
                # Close existing position first
                if current_signal != 0:
                    # Calculate position values at current prices
                    value1 = position_asset1 * prices1[t]
                    value2 = position_asset2 * prices2[t]
                    
                    # Commission on closing (based on absolute values)
                    commission = (abs(value1) + abs(value2)) * self.commission_rate
                    
                    # Settle positions: reverse the cash flows
                    # If we had bought (positive shares), we sell and get cash
                    # If we had sold (negative shares), we buy back and pay cash
                    cash += value1 + value2 - commission
                    
                    # Borrow costs for the holding period (already applied daily, this is final settlement)
                    
                    # Record trade
                    self.trades.append({
                        'date': self.data.index[t],
                        'action': 'CLOSE',
                        'signal': current_signal,
                        'asset1_shares': position_asset1,
                        'asset2_shares': position_asset2,
                        'asset1_price': prices1[t],
                        'asset2_price': prices2[t],
                        'pnl': value1 + value2,
                        'commission': commission,
                        'cash': cash
                    })
                    
                    position_asset1 = 0
                    position_asset2 = 0
                
                # Open new position
                
        # === Abrir nueva posición ===
                if signal != 0:
                   equity = cash + position_asset1 * prices1[t] + position_asset2 * prices2[t]
                   capital_to_use = equity * self.position_pct
                   
                   beta_eff = float(abs(beta)) if beta != 0 else 1.0
                   s = capital_to_use / (abs(prices1[t]) + beta_eff * abs(prices2[t]))
                   
                   if signal == 1:  # Long spread
                       position_asset1 =  s
                       position_asset2 = -beta_eff * s
                   else:             # Short spread
                       position_asset1 = -s
                       position_asset2 =  beta_eff * s

                    # Comisión de apertura
                   open_value1 = abs(position_asset1 * prices1[t])
                   open_value2 = abs(position_asset2 * prices2[t])
                   commission = (open_value1 + open_value2) * self.commission_rate
                   cash -= commission

                   self.trades.append({
                       'date': self.data.index[t],
                       'action': 'OPEN',
                       'signal': signal,
                       'asset1_shares': position_asset1,
                       'asset2_shares': position_asset2,
                       'asset1_price': prices1[t],
                       'asset2_price': prices2[t],
                       'commission': commission,
                       'cash': cash
                    })

                current_signal = signal
            
            # Apply borrow costs daily if holding short positions
            if current_signal != 0:
                daily_borrow = 0
                if position_asset1 < 0:
                    daily_borrow += abs(position_asset1 * prices1[t]) * self.borrow_rate
                if position_asset2 < 0:
                    daily_borrow += abs(position_asset2 * prices2[t]) * self.borrow_rate
                cash -= daily_borrow
            
            # Calculate portfolio value
            position_value = position_asset1 * prices1[t] + position_asset2 * prices2[t]
            portfolio_values[t] = cash + position_value
        
        # Store results
        self.portfolio_value = portfolio_values
        self.returns = pd.Series(portfolio_values).pct_change().fillna(0).values
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': self.data.index,
            'Portfolio_Value': portfolio_values,
            'Returns': self.returns,
            'Signal': self.signals
        })
        results_df.set_index('Date', inplace=True)
        
        print(f"\n✓ Backtest complete")
        print(f"  Total trades: {len(self.trades)}")
        print(f"  Final portfolio value: ${portfolio_values[-1]:,.2f}")
        print(f"  Total return: {(portfolio_values[-1]/self.initial_capital - 1)*100:.2f}%")
        
        return results_df
    
    def calculate_metrics(self):

        if self.portfolio_value is None:
            raise ValueError("Run backtest first!")
        
        # Calculate returns
        total_return = (self.portfolio_value[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        n_years = len(self.portfolio_value) / 252
        annualized_return = ((self.portfolio_value[-1] / self.initial_capital) ** (1/n_years) - 1) * 100
        
        # Volatility
        daily_returns = pd.Series(self.returns)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (daily_returns.mean() * 252 / downside_std) if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = pd.Series(self.portfolio_value)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar Ratio
        calmar = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Win rate
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0:
            close_trades = trades_df[trades_df['action'] == 'CLOSE']
            if len(close_trades) > 1:
                trade_returns = close_trades['cash'].diff().dropna()
                win_rate = (trade_returns > 0).sum() / len(trade_returns) * 100
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        # Total commissions and borrow costs
        total_commissions = sum([t['commission'] for t in self.trades])
        
        self.metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)': win_rate,
            'Total Trades': len(self.trades),
            'Total Commissions ($)': total_commissions,
            'Final Value ($)': self.portfolio_value[-1]
        }
        
        return self.metrics
    
    def print_metrics(self):
        """Print performance metrics in formatted table."""
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS")
        print("="*70)
        
        for key, value in self.metrics.items():
            if '$' in key:
                print(f"{key:.<50} ${value:>15,.2f}")
            elif '%' in key:
                print(f"{key:.<50} {value:>15.2f}%")
            elif 'Ratio' in key:
                print(f"{key:.<50} {value:>18.3f}")
            else:
                print(f"{key:.<50} {value:>18,.0f}")
        
        print("="*70)
    
    def plot_results(self, save_path=None):

        if self.portfolio_value is None:
            raise ValueError("Run backtest first!")
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        dates = self.data.index
        
        # Plot 1: Portfolio Value
        axes[0].plot(dates, self.portfolio_value, 'b-', linewidth=2)
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', 
                       label='Initial Capital', alpha=0.5)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
        
        # Plot 2: Drawdown
        cumulative = pd.Series(self.portfolio_value, index=dates)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max * 100
        
        axes[1].fill_between(dates, 0, drawdown, color='red', alpha=0.3)
        axes[1].plot(dates, drawdown, 'r-', linewidth=1.5)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Returns Distribution
        returns_pct = pd.Series(self.returns) * 100
        axes[2].hist(returns_pct, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[2].axvline(x=returns_pct.mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {returns_pct.mean():.3f}%')
        axes[2].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Daily Return (%)', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Backtest results saved to: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Backtester Module")
    print("Run from strategy.py or create custom backtest")