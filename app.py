import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

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
# Custom CSS for a completely new look
# ==========================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(180deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    /* Container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Hero Header (Deep Blue to Teal Gradient) */
    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 2.5rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.25);
        border: 1px solid #14b8a6;
    }
    .hero h1 {
        color: white !important;
        margin: 0 0 0.5rem 0;
        font-size: 2.4rem;
        font-weight: 800;
    }
    .hero p {
        color: #ccfbf1;
        margin: 0;
        font-size: 1.1rem;
    }
    
    /* Price Box (Teal Gradient) */
    .price-box {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(20, 184, 166, 0.3);
        margin: 1.5rem 0;
        border: 1px solid #5eead4;
    }
    .price-label {
        font-size: 0.95rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .price-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Primary Button (Deep Blue) */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        border: none;
        padding: 0.85rem 1rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.5);
        color: white;
    }
    
    /* Styling the sidebar slightly */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Metric values */
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748b;
    }

    /* Customizing the native Streamlit bordered container */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white !important;
        border-radius: 16px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
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
st.sidebar.subheader("📅 Journey Date (2019)")

journey_month = st.sidebar.number_input(
    "Journey Month (1-12)", 
    min_value=1, 
    max_value=12, 
    value=6,
    step=1,
    help="The ML model was trained on 2019 data. Select the month (e.g., 6 for June)."
)

journey_day = st.sidebar.number_input(
    "Journey Day (1-31)", 
    min_value=1, 
    max_value=31, 
    value=15,
    step=1,
    help="Select the day of the month."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🕐 Flight Times")
dep_time = st.sidebar.time_input("Departure time", value=pd.to_datetime("10:30").time())
arr_time = st.sidebar.time_input("Arrival time", value=pd.to_datetime("14:45").time())

# ---- Automatic Duration Calculation ----
dep_datetime = datetime.combine(datetime.today(), dep_time)
arr_datetime = datetime.combine(datetime.today(), arr_time)

if arr_datetime < dep_datetime:
    arr_datetime += timedelta(days=1)

duration_minutes = int((arr_datetime - dep_datetime).total_seconds() / 60)

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

    row["Journey_Day"] = journey_day
    row["Journey_Month"] = journey_month
    row["Dep_Hour"] = dep_time.hour
    row["Dep_Min"] = dep_time.minute
    row["Arrival_Hour"] = arr_time.hour
    row["Arrival_Min"] = arr_time.minute
    row["Duration"] = duration_minutes

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
# Main layout: two columns
# ==========================================
left, right = st.columns([1, 1], gap="large")

with left:
    # Native streamlit container with border=True (acts as our card)
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; color:#1e293b; font-size:1.15rem;'>📋 Flight Summary</h3>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        c1.metric("Airline", airline)
        c2.metric("Stops", total_stops)
        c1.metric("Source", source)
        c2.metric("Destination", destination)
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        formatted_date = f"{month_names[journey_month - 1]} {journey_day}, 2019"
        
        c1.metric("Journey Date", formatted_date)
        c2.metric("Duration", f"{duration_minutes} min")
        c1.metric("Departure", dep_time.strftime("%H:%M"))
        c2.metric("Arrival", arr_time.strftime("%H:%M"))

with right:
    # Native streamlit container with border=True
    with st.container(border=True):
        st.markdown("<h3 style='margin-top:0; color:#1e293b; font-size:1.15rem;'>🔮 Predict Ticket Price</h3>", unsafe_allow_html=True)
        st.write("Confirm your details and click the button below.")
        st.write("") # Small spacer

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

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.caption("Built with Streamlit + Scikit-learn | Flight Price Prediction Project")
