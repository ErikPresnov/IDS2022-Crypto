import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso, Ridge
import functions
import seaborn as sns

if __name__ == "__main__":
    daily = pd.read_csv("BTC-Daily.csv")
    daily = daily.loc[::-1].reset_index(drop=True)
    daily = daily.drop(["unix", "symbol", 'date'], axis=1)
    daily['open'] = round(daily["open"], 3)
    daily['close'] = round(daily["close"], 3)
    daily["MA200"] = functions.calc_MA(daily, 200)
    daily["MA100"] = functions.calc_MA(daily, 100)
    daily["MA50"] = functions.calc_MA(daily, 50)
    daily["MA20"] = functions.calc_MA(daily, 20)
    daily["BBUpper"], daily["BBLower"] = functions.calc_BB(daily)
    rsis = functions.calc_RSI(daily, 14)
    daily = daily[0:daily.shape[0] - 15]
    daily['RSI'] = rsis
    Y = daily["close"]
    X = daily.drop("close", axis=1)

    trainX, testX = X[0:2400], X[2400:]
    trainY, testY = Y[0:2400], Y[2400:]

    ridgeParams = functions.findRidgeParams(trainX, trainY, testX, testY)#[0.0, True, 1000, 'svd', 0.01]
    ridge = Ridge(alpha=0.0, fit_intercept=True, max_iter=1000, solver='svd', tol=0.01)
    ridge.fit(trainX, trainY)
    pred1 = ridge.predict(testX)

    lassoParams = functions.findLassoParams(trainX, trainY, testX, testY)#[0.0, True, 5000, 0.0001]
    lasso = Lasso(alpha=0.0, fit_intercept=True, max_iter=5000, tol=0.0001)
    lasso.fit(trainX, trainY)
    pred2 = lasso.predict(testX)

    testY = testY.reset_index(drop=True)
    sns.lineplot(testY, color="green")
    sns.lineplot(pred1, color="red")
    sns.lineplot(pred2, color="blue")
    plt.show()