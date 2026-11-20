import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime, timedelta

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
    layout="wide",
)

# ==========================================
# Custom CSS for a nicer look
# ==========================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.3);
    }
    .hero h1 {
        color: white !important;
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-size: 1rem;
    }
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid #eef2f7;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .price-box {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.35);
        margin: 1rem 0;
    }
    .price-label {
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        font-size: 1.05rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.45);
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Hero header
# ==========================================
st.markdown("""
<div class="hero">
    <h1>✈️ Flight Price Predictor</h1>
    <p>Get an estimated ticket price (in INR) powered by a Multilayer Perceptron (MLP) neural network.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# Sidebar inputs
# ==========================================
st.sidebar.header("🛫 Flight Details")

airline = st.sidebar.selectbox(
    "Airline",
    ["IndiGo", "Air India", "Jet Airways", "SpiceJet",
     "Vistara", "GoAir", "Multiple carriers", "Air Asia"],
)

source = st.sidebar.selectbox(
    "Source",
    ["Banglore", "Kolkata", "Delhi", "Chennai", "Mumbai"],
)

destination = st.sidebar.selectbox(
    "Destination",
    ["New Delhi", "Banglore", "Cochin", "Kolkata", "Delhi", "Hyderabad"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Journey Date")

journey_date = st.sidebar.date_input(
    "Select journey date",
    value=date(2019, 6, 15),
    min_value=date(2019, 1, 1),
    max_value=date(2020, 12, 31),
    format="YYYY-MM-DD",
    help="The model was trained on data from 2019, so pick a date in that range for best results.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🕐 Flight Times")
dep_time = st.sidebar.time_input("Departure time", value=pd.to_datetime("10:30").time())
arr_time = st.sidebar.time_input("Arrival time", value=pd.to_datetime("14:45").time())

# ---- Automatic Duration Calculation ----
dep_datetime = datetime.combine(journey_date, dep_time)
arr_datetime = datetime.combine(journey_date, arr_time)

# Handle overnight flights (arrival is earlier than departure)
if arr_datetime < dep_datetime:
    arr_datetime += timedelta(days=1)

duration_minutes = int((arr_datetime - dep_datetime).total_seconds() / 60)

# Display the calculated duration in the sidebar
st.sidebar.info(f"⏱️ **Calculated Duration:** {duration_minutes} minutes")

st.sidebar.markdown("---")
st.sidebar.subheader("🛑 Stops")

total_stops = st.sidebar.selectbox(
    "Total Stops",
    ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"],
)

# ==========================================
# Build input row
# ==========================================
def build_input_row():
    row = {col: 0 for col in feature_columns}

    # Numeric features
    row["Journey_Day"] = journey_date.day
    row["Journey_Month"] = journey_date.month
    row["Dep_Hour"] = dep_time.hour
    row["Dep_Min"] = dep_time.minute
    row["Arrival_Hour"] = arr_time.hour
    row["Arrival_Min"] = arr_time.minute
    row["Duration"] = duration_minutes # Now calculated automatically!

    # One-hot categorical features
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
# Main layout: two columns (summary | prediction)
# ==========================================
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Flight Summary</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Airline", airline)
    c2.metric("Stops", total_stops)
    c1.metric("Source", source)
    c2.metric("Destination", destination)
    c1.metric("Journey Date", journey_date.strftime("%d %b %Y"))
    c2.metric("Duration", f"{duration_minutes} min")
    c1.metric("Departure", dep_time.strftime("%H:%M"))
    c2.metric("Arrival", arr_time.strftime("%H:%M"))

    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔮 Predict Ticket Price</div>', unsafe_allow_html=True)
    st.write("Confirm your details and click the button below.")

    if st.button("Predict Price", type="primary"):
        input_df = build_input_row()
        input_scaled = scaler.transform(input_df)
        prediction_normalized = model.predict(input_scaled)[0]
        predicted_price = prediction_normalized * (Y_MAX - Y_MIN) + Y_MIN

        st.markdown(f"""
        <div class="price-box">
            <div class="price-label">Estimated Flight Price</div>
            <div class="price-value">₹ {predicted_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is a machine-learning estimate. Actual prices vary based on "
            "real-time demand, availability, and airline policies."
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn | Flight Price Prediction Project")
