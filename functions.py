from sklearn.linear_model import Lasso, Ridge

alpha = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
fit_inter = [True, False]
max_iters = [1000, 2500, 5000, 7500, 10000, 15000, 25000]
solvers = ["svd", 'cholesky', 'lsqr', 'sparse_cg', 'saga', 'sag']
tolerance = [1e-2, 1e-3, 1e-4, 1e-5]
best_acc_r = 0
err_coef = 0.025


# Method to calculate MA (moving average) -> mean of N prices
def calc_MA(df, n):
    MA = []
    prices = df['open']
    for i in range(len(prices)):
        if i == 0:
            MA.append(0)
        elif i <= n :
            MA.append(prices[:i].mean())
        else:
            MA.append(prices[i-n:i].mean())
    return MA


# Method to calculate Bollinber Bands -> MA +- 2*STD
def calc_BB(df):
    upperBand = []
    lowerBand = []
    MA20 = df["MA20"]
    STD20 = []
    prices = df["open"]
    for i in range(len(prices)):
        if i == 0 or i == 1:
            STD20.append(0)
        elif i <= 20:
            STD20.append(prices[:i].std())
        else:
            STD20.append(prices[i-20:i].std())
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

    for i in range(len(df['open']) - 1):
        price = df['open'][i]
        if i == len(df['open']) - 1:
            window.append(price)
            continue
        difference = round(df['open'][i + 1] - price, 2)
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
        if len(window) < periods:
            window.append(price)
            continue

        if len(window) == periods:
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

    result = [0]*15
    for rs in rsis:
        result.append(rs)
    return result


def findRidgeParams(trainX, trainY, testX, testY):
    best_acc_r = 0
    best_result_r = []
    for a in alpha:
        for b in fit_inter:
            for c in max_iters:
                for d in solvers:
                    for e in tolerance:
                        RidgeReg = Ridge(alpha=a,
                                         fit_intercept=b,
                                         max_iter=c,
                                         solver=d,
                                         tol=e).fit(trainX, trainY)
                        r_predictions = RidgeReg.predict(testX)
                        acc = 0
                        for index, prediction in enumerate(r_predictions):
                            if testY.iloc[index] * (1 + err_coef) > prediction > testY.iloc[index] * (1 - err_coef):
                                acc += 1
                        acc = round(acc / len(testY), 3)
                        if acc > best_acc_r:
                            print("Found new best acc (Ridge) -> " + str(acc))
                            best_acc_r = acc
                            best_result_r = [a, b, c, d, e]
    return best_result_r


def findLassoParams(trainX, trainY, testX, testY):
    best_acc_l = 0
    best_result_l = []
    for a in alpha:
        for b in fit_inter:
            for c in max_iters:
                for e in tolerance:
                    LassoReg = Lasso(alpha=a,
                                     fit_intercept=b,
                                     max_iter=c,
                                     tol=e).fit(trainX, trainY)
                l_predictions = LassoReg.predict(testX)
                acc = 0
                for index, prediction in enumerate(l_predictions):
                    if testY.iloc[index] * (1 + err_coef) > prediction > testY.iloc[index] * (1 - err_coef):
                        acc += 1
                acc = round(acc / len(testY), 3)
                if acc > best_acc_l:
                    print("Found new best acc (Lasso) -> " + str(acc))
                    best_acc_l = acc
                    best_result_l = [a, b, c, e]
    return best_result_l
