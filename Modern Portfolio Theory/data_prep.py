import numpy as np
import yfinance as yf

from data_prep import trading_days

def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    log_returns = np.log(data / data.shift(1)).dropna()
    mean_returns_annual = log_returns.mean() * trading_days
    cov_matrix_annual = log_returns.cov() * trading_days


    return mean_returns_annual, cov_matrix_annual
