import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_results(df, actual, train_pred, test_pred, time_step):
    plt.figure(figsize=(16, 8))
    sns.set_style("darkgrid")
    plt.plot(df.index[time_step:], actual, label='Actual', linewidth=2)
    plt.plot(df.index[time_step:len(train_pred)+time_step], train_pred, label='Train Predict', linewidth=2)
    plt.plot(df.index[len(train_pred)+(time_step*2)+1:], test_pred, label='Test Predict', linewidth=2)
    plt.title('Stock Price Prediction', fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    plt.legend(fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('stock_data.csv')
    data = df['Close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare the dataset
    time_step = 100
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_model((X_train.shape[1], 1))
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=100,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    train_rmse = np.sqrt(np.mean((train_predict - y_train_actual) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict - y_test_actual) ** 2))
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    # Plot the results
    actual_data = scaler.inverse_transform(scaled_data[time_step:])
    plot_results(df, actual_data, train_predict, test_predict, time_step)

if __name__ == "__main__":
    main()
