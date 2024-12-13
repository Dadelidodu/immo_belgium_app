# Evaluation Report: Model Performance and Metrics

## 1. Model Instantiation

```python

# Define the Neural network

# Build the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

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

def train_model(model, X_train, y_train, X_test, y_test):

    early_stopping = EarlyStopping(patience=150, restore_best_weights=True)

    trained_model = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=350,
        batch_size=2000,
        callbacks=[early_stopping],
        verbose=1)

    return trained_model
```

## 2. Training Metrics

```bash

    Training Metrics

    MAE: *71125.8671875*
    RMSE: *133836.14384220913*
    R²: *0.8269559144973755*
    MAPE: *15.500440206535332*
    sMAPE: *15.261694863585006*

    Testing Metrics

    MAE: *75700.5703125*
    RMSE: *128360.92187814333*
    R²: *0.8356913328170776*
    MAPE: *17.02116282618461*
    sMAPE: *16.900477202572294*
```

## 3. Features and Data Leakage Assessment

### List of Features Used

``` bash
    Subtype of Property
    State of the Building,
    Median Revenue per Commune
    Median Price per Commune
    Livable Space (m2)
    PEB
    Primary Energy Consumption (kWh/m2)
    Surface of the Land (m2)
    Construction Year
    Number of Rooms
    Number of Facades
```

### How Features Were Obtained

Features were came from scraped raw data from immoweb urls and then engineered and combined from multiple sources (Statbel for Mean Price and Revenue per Locality). Steps taken to avoid data leakage : Replacing Median Prices per Locality extracted at first from training set by data from external sources.

## 4. Accuracy Computation Procedure

``` bash
    Split Ratio: *75/25*
    Cross-validation Strategy: *K-fold was not used.*
    Test Set Selection: *Stratifying target set to avoid lone classes of Price.*
```

## 5. Efficiency

### Training Time

Time taken to train the model: *53.24 seconds*

### Inference Time

Time taken for a single prediction: *0.08 seconds*

## 6. Final Dataset Summary

### Dataset Characteristics

``` bash
    Total records: *9558 non-null for Train + 3187 non-null for Test*
    Number of features: *11*
    Target variable: *Price*
```

### Steps Followed

#### Merging:

Raw data from scraping + data from statbel.

#### Cleaning: 

Filling columns with missing values, mapping categorical columns with numerical values, dropping missing values for essential columns, removing outliers, setting values as integers and dropping useless columns

#### Scaling and Encoding: 

Standardizing values with zscore for every feature to avoid outlier interference.

#### Validation: 

Use of test set for validation.
