from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
import streamlit as st

# Define the Neural network

# Build the model

#@st.cache_data
def load_data():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../data/cleaned_dataset.csv')
    data = pd.read_csv(dataset_path)
    return  data

#@st.cache_data
def load_train_test():

    script_dir = os.path.dirname(os.path.abspath(__file__))

    X_train_dataset_path = os.path.join(script_dir, '../data/X_train.csv')
    X_train = pd.read_csv(X_train_dataset_path)

    X_test_dataset_path = os.path.join(script_dir, '../data/X_test.csv')
    X_test = pd.read_csv(X_test_dataset_path)

    y_train_dataset_path = os.path.join(script_dir, '../data/y_train.csv')
    y_train = pd.read_csv(y_train_dataset_path)

    y_test_dataset_path = os.path.join(script_dir, '../data/y_test.csv')
    y_test = pd.read_csv(y_test_dataset_path)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_train_test()

def build_model(X_train):

    model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu', kernel_regularizer=l2(0.05)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.05)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.05)),
    Dense(32, activation='relu', kernel_regularizer=l2(0.05)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.05)),
    Dense(1)]
    )
    
    optimizer = Adam(learning_rate=10**-1.8)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    return model

# Train the model

model = build_model(X_train)

#@st.cache_data
def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(patience=150, restore_best_weights=True)

    # # Set up progress bar
    # progress_bar = st.progress(0)

    # Initialize total epochs
    total_epochs = 350
    batch_size = 2000

    # Training the model
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=total_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # # Update the progress bar at each epoch
    # for epoch in range(total_epochs):
    #     progress_bar.progress((epoch + 1) / total_epochs)

    # Return the trained model
    return model
