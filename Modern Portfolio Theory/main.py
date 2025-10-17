import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco


from data_prep import get_data
from input import tickers, start_date, end_date
from input import risk_free_rate, num_portfolios

mean_returns_annual, cov_matrix_annual = get_data(tickers, start_date, end_date)

portfolio_returns = []
portfolio_volatility = []
portfolio_weights = []
sharpe_ratios = []

for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    portfolio_weights.append(weights)

    returns = np.dot(weights, mean_returns_annual)
    portfolio_returns.append(returns)

    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
    portfolio_volatility.append(volatility)

    sharpe = (returns - risk_free_rate) / volatility
    sharpe_ratios.append(sharpe)

results_data = {'Return': portfolio_returns, 'Volatility': portfolio_volatility, 'Sharpe Ratio': sharpe_ratios}
results_df = pd.DataFrame(results_data)

max_sharpe_portfolio = results_df.iloc[results_df['Sharpe Ratio'].idxmax()]
max_sharpe_weights = portfolio_weights[results_df['Sharpe Ratio'].idxmax()]

min_vol_portfolio = results_df.iloc[results_df['Volatility'].idxmin()]
min_vol_weights = portfolio_weights[results_df['Volatility'].idxmin()]

def portfolio_volatility_func(weights, mean_returns, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def portfolio_return_func(weights, mean_returns, cov_matrix):
    return np.dot(weights, mean_returns)


frontier_volatility_list = []

target_returns = np.linspace(min(mean_returns_annual), max(mean_returns_annual), 100)
num_assets = len(tickers)

args = (mean_returns_annual, cov_matrix_annual)

bounds = tuple((0, 1) for asset in range(num_assets))

initial_guess = num_assets * [1 / num_assets]


for target_return in target_returns:
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: portfolio_return_func(x, mean_returns_annual, cov_matrix_annual) - target_return}
    )

    result = sco.minimize(
        fun=portfolio_volatility_func,
        x0=initial_guess,
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    frontier_volatility_list.append(result['fun'])

print("\n" + "-"*50)
print("Maximum Sharpe Ratio Portfolio")
print("-"*50)
print(f"Expected annual return: {max_sharpe_portfolio['Return']:.2%}")
print(f"Annual volatility (risk): {max_sharpe_portfolio['Volatility']:.2%}")
print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe Ratio']:.2f}")
print("Portfolio weights:")
for i, ticker in enumerate(tickers):
    print(f"  {ticker}: {max_sharpe_weights[i]:.2%}")

print("\n" + "-"*50)
print("Minimum Variance Portfolio")
print("-"*50)
print(f"Expected annual return: {min_vol_portfolio['Return']:.2%}")
print(f"Annual volatility (risk): {min_vol_portfolio['Volatility']:.2%}")
print(f"Sharpe Ratio: {min_vol_portfolio['Sharpe Ratio']:.2f}")
print("Portfolio weights:")
for i, ticker in enumerate(tickers):
    print(f"  {ticker}: {min_vol_weights[i]:.2%}")
print("\n" + "-"*50)

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    results_df['Volatility'],
    results_df['Return'],
    c=results_df['Sharpe Ratio'],
    cmap='viridis',
    marker='o',
    s=20
)

plt.plot(
    frontier_volatility_list,
    target_returns,
    'b--', 
    linewidth=2,
    label='Efficient Frontier'
)

plt.scatter(
    max_sharpe_portfolio['Volatility'],
    max_sharpe_portfolio['Return'],
    marker='*',
    color='red',
    s=300,
    label='Max Sharpe Ratio',
    edgecolors='black'
)

plt.scatter(
    min_vol_portfolio['Volatility'],
    min_vol_portfolio['Return'],
    marker='*',
    color='green',
    s=300,
    label='Min Variance',
    edgecolors='black'
)

plt.title('Efficient Frontier - Markowitz Model', fontsize=18)
plt.xlabel('Volatility (Risk)', fontsize=14)
plt.ylabel('Expected Return', fontsize=14)

plt.colorbar(scatter, label="Sharpe Ratio")
plt.legend(loc='best', fontsize=12)

plt.grid(
    True,
    alpha=0.4
)
plt.show()