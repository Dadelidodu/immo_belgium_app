import os
import pandas as pd
import streamlit as st

# Define the Neural network

# Build the model

@st.cache_data
def load_data():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '../data/cleaned_dataset.csv')
    data = pd.read_csv(dataset_path)
    return  data

@st.cache_data
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
