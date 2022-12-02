import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso

# Method to calculate MA (moving average) -> mean of N prices
def calc_MA(df, n):
    MA = []
    prices = df['open']
    for i in range(len(prices)):
        if i + n < len(prices):
            MA.append(prices[i:i + n].mean())
        else:
            MA.append(prices[i:].mean())
    return MA


# Method to calculate Bollinber Bands -> MA +- 2*STD
def calc_BB(df):
    upperBand = []
    lowerBand = []
    MA20 = df["MA20"]
    STD20 = []
    prices = df["close"]
    for i in range(len(prices)):
        if i + 20 < len(prices):
            STD20.append(prices[i:i + 20].std())
        else:
            STD20.append(prices[i:].std())
        upperBand.append(MA20[i] + 2 * STD20[i])
        lowerBand.append(MA20[i] - 2 * STD20[i])

    return upperBand, lowerBand


def calc_RSI(df, periods=14):
    # Initialize containers for avg. gains and losses
    gains = []
    losses = []
    # Create a container for current lookback prices
    window = []
    # Keeps track of previous average values
    prev_avg_gain = None
    prev_avg_loss = None
    rsis = []

    for i, price in enumerate(df['close']):
        if i == 0:
            window.append(price)
            continue
        difference = round(df['close'][i-1] - price, 2)
        if difference > 0:
            gain = difference
            loss = 0
        elif difference < 0:
            gain = 0
            loss = abs(difference)
        else:
            gain = 0
            loss = 0

        gains.append(gain)
        losses.append(loss)
        if i < periods:
            window.append(price)
            continue

        if i == periods:
            avg_gain = sum(gains) / len(gains)
            avg_loss = sum(losses) / len(losses)
        # Use WSM after initial window-length period
        else:
            avg_gain = (prev_avg_gain * (periods - 1) + gain) / periods
            avg_loss = (prev_avg_loss * (periods - 1) + loss) / periods
            # Keep in memory
        prev_avg_gain = avg_gain
        prev_avg_loss = avg_loss
        # Round for later comparison (optional)
        avg_gain = round(avg_gain, 2)
        avg_loss = round(avg_loss, 2)
        prev_avg_gain = round(prev_avg_gain, 2)
        prev_avg_loss = round(prev_avg_loss, 2)
        rs = round(avg_gain / avg_loss, 2)
        rsi = round(100 - (100 / (1 + rs)), 2)
        rsis.append(rsi)
        window.append(price)
        window.pop(0)
        gains.pop(0)
        losses.pop(0)

    return rsis


if __name__ == "__main__":
    daily = pd.read_csv("BTC-Daily.csv")
    daily = daily.drop("unix", axis=1)
    daily = daily.drop("Volume USD", axis=1)
    daily = daily.drop("symbol", axis=1)
    dates = daily['date']
    daily = daily.drop("date", axis=1)
    daily["MA200"] = calc_MA(daily, 200)
    daily["MA100"] = calc_MA(daily, 100)
    daily["MA50"] = calc_MA(daily, 50)
    daily["MA20"] = calc_MA(daily, 20)
    daily["BBUpper"], daily["BBLower"] = calc_BB(daily)
    daily = daily[0:daily.shape[0] - 2]
    Y = daily["close"]
    X = daily.drop("close", axis=1)

    testX, trainX = X[0:300], X[300:]
    testY, trainY = Y[0:300], Y[300:]

    LassoReg = Lasso(alpha=1.0).fit(trainX, trainY)
    predicted = LassoReg.predict(testX)
    correct = 0
    error = 0.01
    for i,pred in enumerate(predicted):
        if testY[i]*(1 + error) > pred > testY[i]*(1 - error):
            correct += 1

    print(correct/len(predicted))

