import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os


class CointegrationAnalysis:
    
    def __init__(self, data):

        self.data = data
        self.correlation_matrix = None
        self.pairs_results = []
        
    def calculate_correlations(self, method='pearson', window=None):

        if window:
            self.correlation_matrix = self.data.rolling(window).corr().iloc[-len(self.data.columns):]
        else:
            self.correlation_matrix = self.data.corr(method=method)
            
        return self.correlation_matrix
    
    def plot_correlation_matrix(self, save_path=None):

        if self.correlation_matrix is None:
            self.calculate_correlations()
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1
        )
        plt.title('Correlation Matrix - Tech Stocks', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to: {save_path}")
        
        plt.show()
        
    def engle_granger_test(self, asset1, asset2, significance=0.05):

        # Get price series
        y = self.data[asset1].values.reshape(-1, 1)
        x = self.data[asset2].values.reshape(-1, 1)
        
        # OLS Regression
        model = LinearRegression()
        model.fit(x, y)
        
        beta = model.coef_[0][0]
        alpha = model.intercept_[0]
        
        # Calculate residuals
        residuals = y.flatten() - (alpha + beta * x.flatten())
        
        # ADF test on residuals
        adf_result = adfuller(residuals, maxlag=1, regression='c', autolag=None)
        
        is_cointegrated = adf_result[1] < significance
        
        return {
            'asset1': asset1,
            'asset2': asset2,
            'beta': beta,
            'alpha': alpha,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'is_cointegrated': is_cointegrated,
            'residuals': residuals
        }
    
    def johansen_test(self, asset1, asset2, significance=0.05):

        # Prepare data
        data_pair = self.data[[asset1, asset2]].values
        
        # Johansen test
        result = coint_johansen(data_pair, det_order=0, k_ar_diff=1)
        
        # Get significance level index
        sig_level_map = {0.1: 0, 0.05: 1, 0.01: 2}
        sig_idx = sig_level_map.get(significance, 1)
        
        # Trace statistic (tests for at least 1 cointegration relationship)
        trace_stat = result.lr1[0]
        trace_crit = result.cvt[0, sig_idx]
        
        # Max eigenvalue statistic (tests for exactly 1 relationship)
        max_eig_stat = result.lr2[0]
        max_eig_crit = result.cvm[0, sig_idx]
        
        # Check if cointegrated
        is_cointegrated = (trace_stat > trace_crit) or (max_eig_stat > max_eig_crit)
        
        return {
            'asset1': asset1,
            'asset2': asset2,
            'trace_statistic': trace_stat,
            'trace_critical_value': trace_crit,
            'max_eigenvalue_statistic': max_eig_stat,
            'max_eigenvalue_critical_value': max_eig_crit,
            'is_cointegrated': is_cointegrated,
            'eigenvectors': result.evec[:, 0],  # First eigenvector
            'eigenvalues': result.eig
        }
    
    def test_all_pairs(self, corr_threshold=0.7, significance=0.05):

        tickers = self.data.columns.tolist()
        
        # Calculate correlations if not done
        if self.correlation_matrix is None:
            self.calculate_correlations()
        
        results = []
        
        print(f"\n{'='*70}")
        print("TESTING ALL PAIRS FOR COINTEGRATION")
        print(f"{'='*70}")
        
        # Test all combinations
        for asset1, asset2 in combinations(tickers, 2):
            correlation = self.correlation_matrix.loc[asset1, asset2]
            
            print(f"\nTesting: {asset1} - {asset2}")
            print(f"Correlation: {correlation:.4f}")
            
            # Only test if correlation is high enough
            if abs(correlation) >= corr_threshold:
                # Engle-Granger test
                eg_result = self.engle_granger_test(asset1, asset2, significance)
                
                # Johansen test
                joh_result = self.johansen_test(asset1, asset2, significance)
                
                # Combine results
                result = {
                    'Pair': f"{asset1}-{asset2}",
                    'Asset1': asset1,
                    'Asset2': asset2,
                    'Correlation': correlation,
                    'EG_Beta': eg_result['beta'],
                    'EG_ADF_Stat': eg_result['adf_statistic'],
                    'EG_ADF_PValue': eg_result['adf_pvalue'],
                    'EG_Cointegrated': eg_result['is_cointegrated'],
                    'JOH_Trace_Stat': joh_result['trace_statistic'],
                    'JOH_Trace_Crit': joh_result['trace_critical_value'],
                    'JOH_MaxEig_Stat': joh_result['max_eigenvalue_statistic'],
                    'JOH_MaxEig_Crit': joh_result['max_eigenvalue_critical_value'],
                    'JOH_Cointegrated': joh_result['is_cointegrated'],
                    'Both_Cointegrated': eg_result['is_cointegrated'] and joh_result['is_cointegrated']
                }
                
                results.append(result)
                
                # Print results
                print(f"  Engle-Granger: {'✓ Cointegrated' if eg_result['is_cointegrated'] else '✗ Not cointegrated'}")
                print(f"    - ADF p-value: {eg_result['adf_pvalue']:.4f}")
                print(f"  Johansen: {'✓ Cointegrated' if joh_result['is_cointegrated'] else '✗ Not cointegrated'}")
                print(f"    - Trace stat: {joh_result['trace_statistic']:.2f} (crit: {joh_result['trace_critical_value']:.2f})")
            else:
                print(f"  Skipped (correlation too low: {correlation:.4f} < {corr_threshold})")
        
        self.pairs_results = pd.DataFrame(results)
        
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Pairs tested: {len(results)}")
        print(f"Cointegrated (Engle-Granger): {self.pairs_results['EG_Cointegrated'].sum()}")
        print(f"Cointegrated (Johansen): {self.pairs_results['JOH_Cointegrated'].sum()}")
        print(f"Cointegrated (Both): {self.pairs_results['Both_Cointegrated'].sum()}")
        
        return self.pairs_results
    
    def get_best_pairs(self, top_n=3, method='both'):

        if self.pairs_results is None or len(self.pairs_results) == 0:
            raise ValueError("No pairs tested yet. Run test_all_pairs() first.")
        
        # Filter based on method
        if method == 'eg':
            filtered = self.pairs_results[self.pairs_results['EG_Cointegrated']]
            filtered = filtered.sort_values('EG_ADF_PValue')
        elif method == 'johansen':
            filtered = self.pairs_results[self.pairs_results['JOH_Cointegrated']]
            filtered = filtered.sort_values('JOH_Trace_Stat', ascending=False)
        else:  # both
            filtered = self.pairs_results[self.pairs_results['Both_Cointegrated']]
            # Sort by combination of metrics
            filtered = filtered.sort_values(['EG_ADF_PValue', 'JOH_Trace_Stat'], 
                                           ascending=[True, False])
        
        return filtered.head(top_n)
    
    def save_results(self, filepath='results/tables/cointegration_results.csv'):

        if self.pairs_results is None or len(self.pairs_results) == 0:
            raise ValueError("No results to save. Run test_all_pairs() first.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.pairs_results.to_csv(filepath, index=False)
        print(f"\nResults saved to: {filepath}")


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    print("Loading data...")
    loader = DataLoader(['AAPL', 'MSFT', 'GOOGL', 'META'])
    
    # Try to load existing data
    try:
        import glob
        csv_files = glob.glob('data/raw/*.csv')
        if csv_files:
            prices = loader.load_data(csv_files[0])
        else:
            prices = loader.download_data()
    except:
        prices = loader.download_data()
    
    # Use only training data for pair selection
    train, test, val = loader.split_data()
    
    print("\n" + "="*70)
    print("COINTEGRATION ANALYSIS")
    print("="*70)
    
    # Initialize analysis
    coint_analysis = CointegrationAnalysis(train)
    
    # Calculate and plot correlations
    print("\nCalculating correlations...")
    coint_analysis.calculate_correlations()
    coint_analysis.plot_correlation_matrix(save_path='results/figures/correlation_matrix.png')
    
    # Test all pairs
    results = coint_analysis.test_all_pairs(corr_threshold=0.7, significance=0.05)
    
    # Get best pairs
    print("\n" + "="*70)
    print("BEST COINTEGRATED PAIRS")
    print("="*70)
    best_pairs = coint_analysis.get_best_pairs(top_n=3)
    print(best_pairs[['Pair', 'Correlation', 'EG_ADF_PValue', 'JOH_Trace_Stat', 
                      'EG_Cointegrated', 'JOH_Cointegrated', 'Both_Cointegrated']])
    
    # Save results
    coint_analysis.save_results()
    
    print("\n✅ Cointegration analysis complete!")
    print("Next step: Implement Kalman Filters")