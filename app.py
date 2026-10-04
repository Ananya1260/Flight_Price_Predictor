import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# Load artifacts
# ==========================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("flight_price_mlp.pkl")
    scaler = joblib.load("flight_price_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# ==========================================
# y_min and y_max from notebook
# ==========================================
Y_MIN = 1759
Y_MAX = 23001

# ==========================================
# Page config
# ==========================================
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Price Predictor")
st.markdown(
    "Enter flight details below to get an **estimated ticket price** (in INR), "
    "powered by a Multilayer Perceptron (MLP) neural network."
)
st.markdown("---")

# ==========================================
# Sidebar inputs
# ==========================================
st.sidebar.header("Flight Details")

airline = st.sidebar.selectbox(
    "Airline",
    ["IndiGo", "Air India", "Jet Airways", "SpiceJet",
     "Vistara", "GoAir", "Multiple carriers", "Air Asia"]
)

source = st.sidebar.selectbox(
    "Source",
    ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"]
)

destination = st.sidebar.selectbox(
    "Destination",
    ["New Delhi", "Banglore", "Cochin", "Kolkata", "Delhi", "Hyderabad"]
)

journey_day = st.sidebar.slider("Journey Day", 1, 31, 15)
journey_month = st.sidebar.slider("Journey Month", 1, 12, 6)

dep_hour = st.sidebar.slider("Departure Hour", 0, 23, 10)
dep_min = st.sidebar.slider("Departure Minute", 0, 59, 30)

arrival_hour = st.sidebar.slider("Arrival Hour", 0, 23, 14)
arrival_min = st.sidebar.slider("Arrival Minute", 0, 59, 45)

duration_minutes = st.sidebar.slider("Duration (minutes)", 60, 1500, 300)

total_stops = st.sidebar.selectbox(
    "Total Stops",
    ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
)

# ==========================================
# Build input row
# ==========================================
def build_input_row():
    # Start with all zeros — matches feature_columns exactly
    row = {col: 0 for col in feature_columns}

    # Numeric features (always present)
    row["Journey_Day"] = journey_day
    row["Journey_Month"] = journey_month
    row["Dep_Hour"] = dep_hour
    row["Dep_Min"] = dep_min
    row["Arrival_Hour"] = arrival_hour
    row["Arrival_Min"] = arrival_min
    row["Duration"] = duration_minutes

    # One-hot categorical features (only set if the column exists)
    for prefix, value in [
        ("Airline_", airline),
        ("Source_", source),
        ("Destination_", destination),
        ("Total_Stops_", total_stops),
    ]:
        col = f"{prefix}{value}"
        if col in row:
            row[col] = 1

    return pd.DataFrame([row])[feature_columns]

# ==========================================
# Prediction
# ==========================================
if st.button("🔮 Predict Price", type="primary"):
    input_df = build_input_row()
    input_scaled = scaler.transform(input_df)
    prediction_normalized = model.predict(input_scaled)[0]
    predicted_price = prediction_normalized * (Y_MAX - Y_MIN) + Y_MIN

    st.markdown("---")
    st.success(f"### 💰 Estimated Flight Price: ₹ {predicted_price:,.0f}")
    st.caption(
        "This is a machine-learning estimate. Actual prices vary based on "
        "real-time demand, availability, and airline policies."
    )

    with st.expander("📋 View entered flight details"):
        st.write(f"- **Airline:** {airline}")
        st.write(f"- **Source:** {source}")
        st.write(f"- **Destination:** {destination}")
        st.write(f"- **Journey:** Day {journey_day}, Month {journey_month}")
        st.write(f"- **Departure:** {dep_hour:02d}:{dep_min:02d}")
        st.write(f"- **Arrival:** {arrival_hour:02d}:{arrival_min:02d}")
        st.write(f"- **Duration:** {duration_minutes} min")
        st.write(f"- **Total stops:** {total_stops}")

st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn | Flight Price Prediction Project")