import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class DataLoader:
    
    def __init__(self, tickers, start_date=None, end_date=None):
        self.tickers = tickers
        
        
        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date
            
        if start_date is None:
            start = datetime.now() - timedelta(days=15*365)
            self.start_date = start.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date
            
        self.data = None
        
    def download_data(self, save=True):
        print(f"Downloading data for {', '.join(self.tickers)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        
       
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=True
        )
        
       
        if len(self.tickers) > 1:
            prices = data['Adj Close'].copy()
        else:
            prices = pd.DataFrame(data['Adj Close'])
            prices.columns = self.tickers
            
        
        prices = prices.dropna()
        
        self.data = prices
        
        print(f"\nData downloaded successfully!")
        print(f"Shape: {prices.shape}")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"Missing values: {prices.isnull().sum().sum()}")
        
        if save:
            self._save_data()
            
        return prices
    
    def _save_data(self):
        os.makedirs('data/raw', exist_ok=True)
        
        filename = f"data/raw/prices_{self.start_date}_to_{self.end_date}.csv"
        self.data.to_csv(filename)
        print(f"\nData saved to: {filename}")
        
    def load_data(self, filename):

        self.data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Data loaded from: {filename}")
        print(f"Shape: {self.data.shape}")
        return self.data
    
    def get_returns(self):

        if self.data is None:
            raise ValueError("No data available. Download or load data first.")
            
        returns = np.log(self.data / self.data.shift(1)).dropna()
        return returns
    
    def split_data(self, train_ratio=0.6, test_ratio=0.2, val_ratio=0.2):

        if self.data is None:
            raise ValueError("No data available. Download or load data first.")
            
        assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, \
            
        n = len(self.data)
        train_end = int(n * train_ratio)
        test_end = train_end + int(n * test_ratio)
        
        train_data = self.data.iloc[:train_end]
        test_data = self.data.iloc[train_end:test_end]
        val_data = self.data.iloc[test_end:]
        
        print("\nData Split:")
        print(f"Training:   {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} days)")
        print(f"Testing:    {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} days)")
        print(f"Validation: {val_data.index[0]} to {val_data.index[-1]} ({len(val_data)} days)")
        
        return train_data, test_data, val_data
    
    def get_summary_statistics(self):

        if self.data is None:
            raise ValueError("No data available. Download or load data first.")
            
        returns = self.get_returns()
        
        summary = pd.DataFrame({
            'Mean Price': self.data.mean(),
            'Std Price': self.data.std(),
            'Min Price': self.data.min(),
            'Max Price': self.data.max(),
            'Mean Return': returns.mean(),
            'Std Return': returns.std(),
            'Sharpe (ann)': returns.mean() / returns.std() * np.sqrt(252)
        })
        
        return summary


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META']
    
    loader = DataLoader(tickers)
    
    prices = loader.download_data(save=True)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(loader.get_summary_statistics())
    
    train, test, val = loader.split_data()
    
    print("\n✅ Data download complete!")
