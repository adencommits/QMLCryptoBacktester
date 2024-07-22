import pandas as pd

# Load the data
data = pd.read_csv("trades.csv")

# Initialize variables for P/L calculation
total_buy_cost = 0
total_sell_value = 0

# Loop through the data to calculate total buy cost and total sell value
for index, row in data.iterrows():
    if row['action'] == 'buy':
        total_buy_cost += row['price'] * row['quantity']
    elif row['action'] == 'sell':
        total_sell_value += row['price'] * row['quantity']

# Calculate total P/L
total_pl = total_sell_value - total_buy_cost

print(f"Total Profit/Loss (P/L): {total_pl:.2f}")
