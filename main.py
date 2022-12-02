import pandas as pd
import numpy as np

# Method to calculate MA (moving average) -> mean of N prices
def calc_MA(df, n):
    MA = []
    prices = df['open']
    for i in range(len(prices)):
        if i + n < len(prices):
            MA.append(prices[i:i+n].mean())
        else:
            MA.append(prices[i:].mean())
    return MA

def calc_BB(df):
    pass

def calc_RSI(df):
    pass

if __name__ == "__main__":
    daily = pd.read_csv("BTC-Daily.csv")
    daily = daily.drop("unix", axis=1)
    daily["MA200"] = calc_MA(daily, 200)
    daily["MA100"] = calc_MA(daily, 100)
    daily["MA50"] = calc_MA(daily, 50)
    print(daily.head(20))
