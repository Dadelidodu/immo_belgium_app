import streamlit as st
from App.Scripts.Map import *
from App.Scripts.Load import *
from tensorflow.keras.models import load_model
from keras import metrics
from tensorflow.keras.utils import get_custom_objects

# Set the layout of the App

st.set_page_config(page_title="Immo Belgium App", page_icon="🏠", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>🏛️ Immo Belgium App 🏛️</h1>", unsafe_allow_html=True
)

# Load the datasets using the cached function

_df = load_geo()
data = load_data()
X_train, X_test, y_train, y_test = load_train_test()

# Display Prediction Model in column 1

with st.sidebar:

    # Title

    st.markdown(
    "<h1 style='text-align: center;'>Price Prediction Tool</h1>", unsafe_allow_html=True
)
    # User inputs

    zip_code = st.sidebar.selectbox(
        "🗺️ Select Zip Code",
        options=X_train["Zip Code"].sort_values(ascending=True).unique(),
    )
    prop_type = st.sidebar.selectbox(
        "🏠 Select Type of Property", options=X_train["Subtype of Property"].unique()
    )
    livable_space_score = st.sidebar.number_input(
        "📐 Enter Livable Space (m2)", min_value=0, step=10
    )
    land_surface_score = st.sidebar.number_input(
        "🌳 Enter Surface of the Land (m2)", min_value=0, step=10
    )
    energy_consumption_score = st.sidebar.number_input(
        "⚡ Enter Primary Energy Consumption (kWh/m2)", min_value=0, step=10
    )
    construction_year_score = st.sidebar.number_input(
        "📅 Enter Construction Year", min_value=1750, step=10
    )
    facades_score = st.sidebar.number_input(
        "🏠 Enter Number of Facades", min_value=0, step=1, max_value=4
    )
    rooms_score = st.sidebar.number_input(
        "🛏️ Enter Number of Rooms", min_value=0, step=1, max_value=20
    )
    building_states = [
        "As new",
        "Just renovated",
        "Good",
        "To be done up",
        "To renovate",
        "To restore",
    ]
    building_state_index = st.sidebar.selectbox(
        "🛠️ Select State of the Building",
        options=range(len(building_states)),
        format_func=lambda x: building_states[x],
    )
    building_score = building_state_index
    PEB_rankings = ["A", "B", "C", "D", "E", "F", "G"]
    PEB_index = st.sidebar.selectbox(
        "📝 Select PEB",
        options=range(len(PEB_rankings)),
        format_func=lambda x: PEB_rankings[x],
    )
    PEB_score = PEB_index

    if st.sidebar.button("Predict Price"):

        # Calculate scores for inputs

        median_revenue_score = X_train.loc[
            X_train["Zip Code"] == zip_code, "Median Revenue per Commune"
        ].mean()
        median_price_score = X_train.loc[
            X_train["Zip Code"] == zip_code, "Median Price per Commune"
        ].mean()
        prop_type_score = X_train.loc[
            X_train["Subtype of Property"] == prop_type, "Median Revenue per Commune"
        ].mean()
        PEB_type_score = X_train.loc[
            X_train["PEB"] == PEB_score, "Median Revenue per Commune"
        ].mean()
        building_state_type_score = X_train.loc[
            X_train["State of the Building"] == building_score,
            "Median Revenue per Commune",
        ].mean()

        # Standardize Inputs

        input_data = {
            "Median Revenue per Commune": median_revenue_score,
            "Median Price per Commune": median_price_score,
            "Livable Space (m2)": livable_space_score,
            "Subtype of Property Score": prop_type_score,
            "State of the Building Score": building_state_type_score,
            "PEB Score": PEB_type_score,
            "Primary Energy Consumption (kWh/m2)": energy_consumption_score,
            "Surface of the Land (m2)": land_surface_score,
            "Construction Year": construction_year_score,
            "Number of Rooms": rooms_score,
            "Number of Facades": facades_score,
        }

        input_features = pd.DataFrame([input_data])

        for col in input_features.columns:
            input_features[col] = (
                (input_features[col] - X_train[col].mean()) / X_train[col].std()
            ).round(3)
        final_features = input_features.to_numpy()
        print(final_features)

        # Load model

        get_custom_objects().update({"mae": metrics.mean_absolute_error})
        trained_model = load_model("App/data/trained_model.h5")

        # Predict using the trained model
        predicted_price = trained_model.predict(final_features)[0]
        predicted_price_value = round(predicted_price.item(), -4)

        # Display the prediction

        custom_css = """
        <style>
            .predicted-price {
                font-size: 2rem; /* Increase the font size */
                font-weight: bold; /* Make the text bold */
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); /* Add a subtle shadow */
                border: 2px solid #4CAF50; /* Add a border */
                padding: 10px; /* Add some padding */
                border-radius: 10px; /* Round the corners */
                background-color: #333346; /* Set a light green background */
                display: inline-block; /* Inline styling */
            }
        </style>
        """

        # Inject CSS into the app
        st.markdown(custom_css, unsafe_allow_html=True)

        # Display the predicted price with custom styling
        st.markdown(
            f"<div class='predicted-price'>Predicted Price: €{predicted_price_value:,.2f}</div>",
            unsafe_allow_html=True,
        )

# Display Map & Dataset in column 2 for interactivity

st.markdown(
    """
    <style>
    iframe {
        width: 100% !important
    }
    </style>
    """,
    unsafe_allow_html=True,
)

display_map(_df)

st.markdown(
    """
    <style>
    iframe {
        width: 100% !important
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write(data)
