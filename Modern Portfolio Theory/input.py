from datetime import datetime

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2021-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Model Parameters
trading_days = 252
risk_free_rate = 0.04
num_portfolios = 25000 # numer of portfolios to be simulated