import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import datetime
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# Load the saved Random Forest model
model = joblib.load("water_quality_model.pkl")

# Define all possible categories for encoding
all_sources = ["River", "Lake", "Groundwater", "Well"]
all_colors = ["Clear", "Brown", "Yellow", "Green"]
all_odors = ["Odorless", "Foul", "Metallic"]
all_times = ["Morning", "Afternoon", "Evening", "Night"]

# Create LabelEncoders
encoder_source = LabelEncoder().fit(all_sources)
encoder_color = LabelEncoder().fit(all_colors)
encoder_odor = LabelEncoder().fit(all_odors)
encoder_time = LabelEncoder().fit(all_times)

known_sources = list(encoder_source.classes_)
known_colors = list(encoder_color.classes_)
known_odors = list(encoder_odor.classes_)
known_times = list(encoder_time.classes_)

default_ph = 7.453083
default_sulfate = 133.9778
default_water_temperature = 16.46738

def map_unknown(value, known_classes, default="Unknown"):
    return value if value in known_classes else default

# FastAPI endpoint URLs
FASTAPI_SAVE_URL = "http://127.0.0.1:8000/save_predictions/"
FASTAPI_GET_URL = "http://127.0.0.1:8000/get_predictions/"

# Page Configuration
st.set_page_config(page_title="Aqua Predict", layout="wide")
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Go to",
    ("🏠 Home", "🔹 Single Prediction", "📂 Batch Prediction", "📜 View Past Predictions")
)

def convert_month(month):
    if isinstance(month, str):
        try:
            return datetime.datetime.strptime(month, "%B").month
        except ValueError:
            return 1
    return int(month)

def predict_single(data):
    prediction = model.predict([data])
    return prediction[0]

def predict_batch(input_data):
    predictions = model.predict(input_data)
    input_data["Prediction"] = predictions
    return input_data

# ---------- HOME PAGE ----------
if mode == "🏠 Home":
    st.markdown(
        """
        <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        .block-container {
            padding: 2rem 3rem;
        }

        /* Header bar */
        .header-bar {
            background-color: #003366;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        }

        .header-bar h1 {
            color: #ffffff;
            margin: 0;
            font-size: 2.2rem;
            text-align: left;
        }

        .header-bar p {
            color: #d0d0d0;
            margin: 0;
            font-size: 1.1rem;
            text-align: left;
        }

        /* Team section */
        .team-container {
            background-color: #f2f6fc;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 2rem;
        }

        .team-columns {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .team-column {
            flex: 1;
            min-width: 200px;
        }

        .team-column ul {
            padding-left: 20px;
        }
        .team-container li {
        color: #000000;
    }

        /* Sidebar center */
        section[data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="header-bar">
            <h1>🌊 Aqua Predict</h1>
            <p>AI-powered Water Quality Prediction Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
    Welcome to **Aqua Predict** – an AI-powered platform that predicts water quality based on chemical and physical parameters.

    **Features:**
    - 🔹 *Single Prediction:* Enter water sample data manually.
    - 📂 *Batch Prediction:* Upload CSV files for bulk predictions.
    - 📜 *View Past Predictions:* Browse or download your prediction history.
    """)

    st.subheader("👥 Meet the Team")
    st.markdown(
        """
        <div class="team-container">
            <div class="team-columns">
                <div class="team-column">
                    <ul>
                        <li><strong>Sameer Shaik</strong> – Lead Data Scientist</li>
                        <li><strong>Sree Charan</strong> – Backend Developer</li>
                    </ul>
                </div>
                <div class="team-column">
                    <ul>
                        <li><strong>Grishma Reddy</strong> – Frontend Engineer</li>
                        <li><strong>Brahma Reddy</strong> – Frontend Engineer</li>
                        <li><strong>Jayaram Bhanu Prakash Navudu</strong> – Project Manager</li>
                    </ul>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;color:#888;font-size:0.9em">
            © 2025 Aqua Predict Team. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- SINGLE PREDICTION ----------
elif mode == "🔹 Single Prediction":
    st.title("🔹 Single Water Quality Prediction")

    ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
    iron = st.number_input("Iron (mg/L)", step=0.1)
    nitrate = st.number_input("Nitrate (mg/L)", step=0.1)
    chloride = st.number_input("Chloride (mg/L)", step=0.1)
    lead = st.number_input("Lead (mg/L)", step=0.1)
    zinc = st.number_input("Zinc (mg/L)", step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", step=0.1)
    fluoride = st.number_input("Fluoride (mg/L)", step=0.1)
    copper = st.number_input("Copper (mg/L)", step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", step=0.1)
    conductivity = st.number_input("Conductivity (μS/cm)", step=1.0)
    chlorine = st.number_input("Chlorine (mg/L)", step=0.1)
    manganese = st.number_input("Manganese (mg/L)", step=0.1)
    tds = st.number_input("Total Dissolved Solids (mg/L)", step=1.0)
    water_temp = st.number_input("Water Temperature (°C)", step=0.1)
    air_temp = st.number_input("Air Temperature (°C)", step=0.1)
    month = st.slider("Month", 1, 12, 1)
    day = st.slider("Day", 1, 31, 1)

    source = st.selectbox("Source", all_sources)
    color = st.selectbox("Color", all_colors)
    odor = st.selectbox("Odor", all_odors)
    time_of_day = st.selectbox("Time of Day", all_times)

    source_encoded = encoder_source.transform([source])[0]
    color_encoded = encoder_color.transform([color])[0]
    odor_encoded = encoder_odor.transform([odor])[0]
    time_encoded = encoder_time.transform([time_of_day])[0]

    input_data = [
        ph, iron, nitrate, chloride, lead, zinc, color_encoded,
        turbidity, fluoride, copper, odor_encoded, sulfate,
        conductivity, chlorine, manganese, tds, source_encoded,
        water_temp, air_temp, month, day, time_encoded
    ]

    if st.button("💧 Predict Water Quality"):
        prediction = predict_single(input_data)
        st.success(f"Predicted Water Quality Index: {prediction:.2f}")

        payload = {
            "ph": ph, "iron": iron, "nitrate": nitrate, "chloride": chloride, "lead": lead, "zinc": zinc,
            "color": color, "turbidity": turbidity, "fluoride": fluoride, "copper": copper,
            "odor": odor, "sulfate": sulfate, "conductivity": conductivity, "chlorine": chlorine,
            "manganese": manganese, "tds": tds, "source": source, "water_temp": water_temp,
            "air_temp": air_temp, "month": month, "day": day, "time_of_day": time_of_day,
            "prediction": int(round(prediction))
        }

        try:
            response = requests.post(FASTAPI_SAVE_URL, json=[payload])
            if response.status_code == 200:
                st.info("Prediction saved to the database successfully.")
            else:
                st.error("Error saving prediction.")
        except Exception as e:
            st.error(f"Error: {e}")

# ---------- BATCH PREDICTION ----------
elif mode == "📂 Batch Prediction":
    st.title("📂 Batch Water Quality Prediction")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        original_data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(original_data)

        original_data["Source"] = original_data["Source"].fillna("River").apply(lambda x: map_unknown(x, known_sources, "River"))
        original_data["Color"] = original_data["Color"].fillna("Clear").apply(lambda x: map_unknown(x, known_colors, "Clear"))
        original_data["Odor"] = original_data["Odor"].fillna("Odorless").apply(lambda x: map_unknown(x, known_odors, "Odorless"))
        original_data["Time of Day"] = original_data["Time of Day"].fillna("Morning").apply(lambda x: map_unknown(x, known_times, "Morning"))
        original_data["Month"] = original_data["Month"].apply(convert_month)
        original_data["Day"] = original_data["Day"].fillna(1).astype(int)

        input_data = original_data.copy()
        input_data["Source"] = encoder_source.transform(input_data["Source"])
        input_data["Color"] = encoder_color.transform(input_data["Color"])
        input_data["Odor"] = encoder_odor.transform(input_data["Odor"])
        input_data["Time of Day"] = encoder_time.transform(input_data["Time of Day"])
        input_data["pH"] = input_data["pH"].fillna(default_ph)
        input_data["Sulfate"] = input_data["Sulfate"].fillna(default_sulfate)
        input_data["Water Temperature"] = input_data["Water Temperature"].fillna(default_water_temperature)

        if st.button("💧 Predict Batch"):
            results = predict_batch(input_data)
            st.write("Prediction Results:")
            st.dataframe(results)
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results", csv, "predictions.csv", "text/csv")

            batch_payloads = []
            for _, row in results.iterrows():
                payload = {
                    "ph": float(row["pH"]),
                    "iron": float(row["Iron"]),
                    "nitrate": float(row["Nitrate"]),
                    "chloride": float(row["Chloride"]),
                    "lead": float(row["Lead"]),
                    "zinc": float(row["Zinc"]),
                    "color": str(row["Color"]),
                    "turbidity": float(row["Turbidity"]),
                    "fluoride": float(row["Fluoride"]),
                    "copper": float(row["Copper"]),
                    "odor": str(row["Odor"]),
                    "sulfate": float(row["Sulfate"]),
                    "conductivity": float(row["Conductivity"]),
                    "chlorine": float(row["Chlorine"]),
                    "manganese": float(row["Manganese"]),
                    "tds": float(row["Total Dissolved Solids"]),
                    "source": str(row["Source"]),
                    "water_temp": float(row["Water Temperature"]),
                    "air_temp": float(row["Air Temperature"]),
                    "month": int(row["Month"]),
                    "day": int(row["Day"]),
                    "time_of_day": str(row["Time of Day"]),
                    "prediction": int(row["Prediction"])
                }
                batch_payloads.append(payload)

            try:
                response = requests.post(FASTAPI_SAVE_URL, json=batch_payloads)
                if response.status_code == 200:
                    st.info("Batch predictions saved successfully.")
                else:
                    st.error(f"Error saving predictions: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- VIEW PAST PREDICTIONS ----------
elif mode == "📜 View Past Predictions":
    st.title("📜 View Past Predictions")
    st.write("Filter by date range (optional):")
    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    params = {}
    if start:
        params["start_date"] = start.isoformat()
    if end:
        params["end_date"] = end.isoformat()

    if st.button("🔍 Fetch Predictions"):
        try:
            response = requests.get(FASTAPI_GET_URL, params=params)
            if response.status_code == 200:
                predictions = response.json()
                if predictions:
                    st.dataframe(predictions)
                else:
                    st.info("No predictions found.")
            else:
                st.error("Error fetching predictions.")
        except Exception as e:
            st.error(f"Error: {e}")
