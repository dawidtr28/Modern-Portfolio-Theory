import numpy as np
import yfinance as yf

def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    log_returns = np.log(data / data.shift(1)).dropna()
    mean_returns_annual = log_returns.mean() * 252
    cov_matrix_annual = log_returns.cov() * 252

    return mean_returns_annual, cov_matrix_annual