from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import shap

# Define the Neural network

# Build the model


def build_model(X_train):

    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(256, activation="relu", kernel_regularizer=l2(0.05)),
            Dense(128, activation="relu", kernel_regularizer=l2(0.05)),
            Dense(64, activation="relu", kernel_regularizer=l2(0.05)),
            Dense(32, activation="relu", kernel_regularizer=l2(0.05)),
            Dense(16, activation="relu", kernel_regularizer=l2(0.05)),
            Dense(1),
        ]
    )

    optimizer = Adam(learning_rate=10**-1.8)
    model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])

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
        verbose=1,
    )

    return trained_model


# SHAP ANALYSIS


def shap_analysis(X_train, X_test, model):

    background = shap.sample(X_train, 100)
    X_test_sample = shap.sample(X_test, 100)
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(X_test_sample, nsamples=100)

    return background, X_test_sample, explainer, shap_values
