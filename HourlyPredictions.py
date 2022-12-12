import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

if __name__ == "__main__":
    hours = 168  # Lookback period (168 hours/1 week) in our case
    # Read the data from a csv file, reverse the dataframe since the file goes from new to old
    hourly = pd.read_csv("BTC-Hourly.csv")[::-1].reset_index(drop=True)

    data = hourly.filter(['close'])  # Take the closing price
    dataset = data.values

    trainingLen = int(len(dataset) * 0.9)  # Find training data size (90% in this case)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(dataset)  # Scale the data to [0,1] because it is needed for the model

    # Separate data into training and testing datasets
    trainingData, testingData = scaled[0:trainingLen, :], scaled[trainingLen - hours:, :]
    train_X = []
    train_y = []
    for i in range(hours, len(trainingData)):  # Separate training data into X and Y categories
        train_X.append(trainingData[i - hours:i, 0])
        train_y.append(trainingData[i, 0])

    test_X = []
    for i in range(hours, len(testingData)):
        test_X.append(testingData[i - hours:i, 0])

    test_X = np.array(test_X)  # Converto to np.array
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))  # Reshape data since LSTM requires a 3D input [batch, timesteps, feature]

    train_X, train_y = np.array(train_X), np.array(train_y)  # Convert  to np.arrays
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))  # Reshape data since LSTM requires a 3D input [batch, timesteps, feature]

    # Create the model
    model = Sequential()
    model.add(LSTM(168, return_sequences=True, input_shape=(hours, 1)))
    model.add(LSTM(168, return_sequences=False))
    model.add(Dense(168))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")  # Compile the model

    model.fit(train_X, train_y, batch_size=1, epochs=10)  # Train the model

    predictions = model.predict(test_X)
    predictions = scaler.inverse_transform(predictions)

    set1 = data[:trainingLen]
    set1["Training data"] = set1['close']
    set2 = data[trainingLen:]
    set2["Actual"] = set2['close']
    set2 = set2.assign(Predicted=predictions)

    acc = 0
    error = 0.05
    for i, value in enumerate(set2['Actual']):
        predicted = set2['Predicted'].iloc[i]
        if value * (1 - error) < predicted < value * (1 + error):
            acc += 1

    acc = acc / len(set2['Actual'])
    print(acc)

    # Plot the results
    plt.plot(set1["Training data"])
    plt.plot(set2['Actual'])
    plt.plot(set2['Predicted'])
    plt.legend(labels=["Training data", 'Actual', 'Predicted'])
    plt.show()