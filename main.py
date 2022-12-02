import pandas as pd


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


def calc_BB(df):
    pass


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
    daily["MA200"] = calc_MA(daily, 200)
    daily["MA100"] = calc_MA(daily, 100)
    daily["MA50"] = calc_MA(daily, 50)
    # print(daily.head(20))
    print(calc_RSI(daily))
    print(daily['close'])
    print(daily.head(10) )
