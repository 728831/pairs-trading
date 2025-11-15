import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
import os


class KalmanFilterHedgeRatio:
    
    def __init__(self, delta=1e-4, Ve=1e-3):

        self.delta = delta  # Process noise (Q)
        self.Ve = Ve        # Measurement noise (R)
        
        # State estimate
        self.beta = None    # Current hedge ratio estimate
        self.P = None       # Error covariance
        
        # History
        self.beta_history = []
        self.P_history = []
        self.kalman_gain_history = []
        
    def initialize(self, initial_beta=1.0, initial_P=1.0):

        self.beta = initial_beta
        self.P = initial_P
        
        self.beta_history = [initial_beta]
        self.P_history = [initial_P]
        self.kalman_gain_history = []
        
    def predict(self):

        # State prediction (random walk model)
        beta_pred = self.beta
        
        # Covariance prediction
        P_pred = self.P + self.delta
        
        return beta_pred, P_pred
    
    def update(self, y, x):

        # Predict
        beta_pred, P_pred = self.predict()
        
        # Innovation (prediction error)
        innovation = y - beta_pred * x
        
        # Innovation variance
        S = x**2 * P_pred + self.Ve
        
        # Kalman Gain
        K = P_pred * x / S
        
        # State update
        self.beta = beta_pred + K * innovation
        
        # Covariance update
        self.P = (1 - K * x) * P_pred
        
        # Store history
        self.beta_history.append(self.beta)
        self.P_history.append(self.P)
        self.kalman_gain_history.append(K)
        
        return self.beta, K
    
    def filter(self, y_series, x_series, initial_beta=None):

        # Initialize with OLS estimate if not provided
        if initial_beta is None:
            initial_beta = np.cov(y_series, x_series)[0, 1] / np.var(x_series)
            
        self.initialize(initial_beta=initial_beta, initial_P=1.0)
        
        # Run filter
        for y, x in zip(y_series, x_series):
            self.update(y, x)
            
        return (
            np.array(self.beta_history),
            np.array(self.P_history),
            np.array(self.kalman_gain_history)
        )
    
    def get_current_estimate(self):
        return self.beta
    
    def plot_results(self, dates=None, save_path=None):

        betas = np.array(self.beta_history)
        stds = np.sqrt(np.array(self.P_history))
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        x_axis = dates if dates is not None else np.arange(len(betas))
        
        # Plot 1: Hedge Ratio with confidence bands
        axes[0].plot(x_axis, betas, 'b-', label='Beta Estimate', linewidth=2)
        axes[0].fill_between(
            x_axis,
            betas - 2*stds,
            betas + 2*stds,
            alpha=0.3,
            label='95% Confidence'
        )
        axes[0].set_title('Dynamic Hedge Ratio (Kalman Filter)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Hedge Ratio (β)', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Kalman Gain
        if len(self.kalman_gain_history) > 0:
            gains = np.array(self.kalman_gain_history)
            axes[1].plot(x_axis[1:], gains, 'r-', label='Kalman Gain', linewidth=1.5)
            axes[1].set_title('Kalman Gain Evolution', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Kalman Gain', fontsize=12)
            axes[1].set_xlabel('Time', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hedge ratio plot saved to: {save_path}")
        
        plt.show()


class KalmanFilterTradingSignals:

    
    def __init__(self, delta=1e-3, Ve=1e-2):

        self.delta = delta
        self.Ve = Ve
        
        # State
        self.spread_estimate = None
        self.P = None
        
        # History
        self.spread_history = []
        self.P_history = []
        
    def initialize(self, initial_spread=0.0, initial_P=1.0):
        self.spread_estimate = initial_spread
        self.P = initial_P
        
        self.spread_history = [initial_spread]
        self.P_history = [initial_P]
        
    def update(self, observed_spread):

        # Predict
        spread_pred = self.spread_estimate
        P_pred = self.P + self.delta
        
        # Innovation
        innovation = observed_spread - spread_pred
        
        # Kalman Gain
        S = P_pred + self.Ve
        K = P_pred / S
        
        # Update
        self.spread_estimate = spread_pred + K * innovation
        self.P = (1 - K) * P_pred
        
        # Store
        self.spread_history.append(self.spread_estimate)
        self.P_history.append(self.P)
        
        return self.spread_estimate
    
    def filter(self, spread_series):

        self.initialize(initial_spread=spread_series[0])
        
        for spread in spread_series:
            self.update(spread)
            
        return np.array(self.spread_history)


def calculate_spread_vecm(prices1, prices2, beta):

    if isinstance(beta, (int, float)):
        beta = np.full(len(prices1), beta)
        
    spread = prices1 - beta * prices2
    return spread


if __name__ == "__main__":
    print("Kalman Filter Module")
    print("This module provides Kalman Filters for:")
    print("1. Dynamic Hedge Ratio Estimation")
    print("2. Trading Signal Generation")
    print("\nRun strategy.py to see full implementation.")