import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == "__main__":
    # 1. Load your training data
    train_data = pd.read_csv('./data/train_data_RNN.csv')
    test_data = pd.read_csv('./data/test_data_RNN.csv')

    # Extract the features and target variables from the training data
    train_features = train_data.iloc[:, :-1].values
    train_target = train_data.iloc[:, -1].values

    # Extract the features and target variables from the testing data
    test_features = test_data.iloc[:, :-1].values
    test_target = test_data.iloc[:, -1].values

    # Perform feature scaling for training and testing data
    scaler = MinMaxScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_train_target = scaler.fit_transform(train_target.reshape(-1, 1))
    scaled_test_features = scaler.fit_transform(test_features)
    scaled_test_target = scaler.fit_transform(test_target.reshape(-1, 1))

    # Reshape the features for LSTM input [samples, time steps, features]
    reshaped_train_features = np.reshape(scaled_train_features, (scaled_train_features.shape[0], 1, scaled_train_features.shape[1]))
    reshaped_test_features = np.reshape(scaled_test_features, (scaled_test_features.shape[0], 1, scaled_test_features.shape[1]))

    # 2. Train your network
    # Create the RNN model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(1, train_features.shape[1])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Define the model checkpoint callback to save the best model during training
    checkpoint_callback = ModelCheckpoint(filepath='models/20941704.h5', save_best_only=True, save_weights_only=False)

    # Train the model on the training data
    train_history = model.fit(reshaped_train_features, scaled_train_target, epochs=100, batch_size=32, callbacks=[checkpoint_callback])

    # Print the final training loss
    final_train_loss = train_history.history['loss'][-1]
    print(f"Final Training Loss: {final_train_loss}")

    # 3. Save the trained model for training data
    model.save('models/20941704_train.h5')

    # Evaluate the model on the testing data
    test_loss = model.evaluate(reshaped_test_features, scaled_test_target)
    print(f"Test Loss: {test_loss}")

    # 4. Save the trained model for testing data
    model.save('models/20941704_test.h5')
