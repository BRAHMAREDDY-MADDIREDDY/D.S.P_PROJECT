import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved Random Forest model
model = joblib.load('water_quality_model.pkl')

# Define all possible categories for encoding
all_sources = ['River', 'Lake', 'Groundwater', 'Well']
all_colors = ['Clear', 'Brown', 'Yellow', 'Green']
all_odors = ['Odorless', 'Foul', 'Metallic']
all_times = ['Morning', 'Afternoon', 'Evening', 'Night']

# Create separate LabelEncoders for each categorical feature
encoder_source = LabelEncoder().fit(all_sources)
encoder_color = LabelEncoder().fit(all_colors)
encoder_odor = LabelEncoder().fit(all_odors)
encoder_time = LabelEncoder().fit(all_times)

# Page Configuration
st.set_page_config(page_title="Aqua Predict", layout="centered")
st.title("Aqua Predict")

# Function for Single Prediction
def predict_single(data):
    prediction = model.predict([data])
    return prediction[0]

# Function for Batch Prediction
def predict_batch(input_data):
    predictions = model.predict(input_data)
    input_data['Prediction'] = predictions
    return input_data

# Sidebar for Navigation
st.sidebar.header("Choose Prediction Mode")
prediction_mode = st.sidebar.radio("Prediction Mode", ('Single Prediction', 'Batch Prediction (CSV Upload)'))

# Single Prediction Mode
if prediction_mode == 'Single Prediction':
    st.header("Input Water Quality Parameters")

    # Numeric Input Fields
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

    # Categorical Input Fields
    source = st.selectbox("Source", all_sources)
    color = st.selectbox("Color", all_colors)
    odor = st.selectbox("Odor", all_odors)
    time_of_day = st.selectbox("Time of Day", all_times)

    # Encode categorical values using the separate encoders
    source_encoded = encoder_source.transform([source])[0]
    color_encoded = encoder_color.transform([color])[0]
    odor_encoded = encoder_odor.transform([odor])[0]
    time_encoded = encoder_time.transform([time_of_day])[0]

    # Prepare data for prediction in the same order as model training
    # (Assuming your model was trained on 23 features with the following order)
    input_data = [
        ph, iron, nitrate, chloride, lead, zinc, 
        color_encoded, turbidity, fluoride, copper, 
        odor_encoded, sulfate, conductivity, chlorine, 
        manganese, tds, source_encoded, water_temp, 
        air_temp, month, day, time_encoded
    ]

    # Predict button for single prediction
    if st.button("Predict Water Quality"):
        prediction = predict_single(input_data)
        st.success(f"Predicted Water Quality: {prediction:.2f}")

# Batch Prediction Mode
else:
    st.header("Batch Prediction via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        st.write("Preview of Uploaded Data:")
        input_data = pd.read_csv(uploaded_file)
        st.dataframe(input_data.head())

        # Print all column names for debugging
        st.write("Columns in uploaded CSV:")
        st.write(input_data.columns)

        # Ensure that the categorical columns are properly encoded
        try:
            # Mapping unexpected values to a default category
            input_data['Source'] = input_data['Source'].apply(lambda x: x if x in all_sources else 'River')
            input_data['Color'] = input_data['Color'].apply(lambda x: x if x in all_colors else 'Clear')
            input_data['Odor'] = input_data['Odor'].apply(lambda x: x if x in all_odors else 'Odorless')
            input_data['Time of Day'] = input_data['Time of Day'].apply(lambda x: x if x in all_times else 'Morning')
            input_data['Month'] = input_data['Month'].apply(lambda x: x if x.isdigit() else '1')  # Default to January if not a digit

            # Encode categorical columns in batch mode using our defined encoders
            input_data['Source'] = encoder_source.transform(input_data['Source'])
            input_data['Color'] = encoder_color.transform(input_data['Color'])
            input_data['Odor'] = encoder_odor.transform(input_data['Odor'])
            input_data['Time of Day'] = encoder_time.transform(input_data['Time of Day'])
            input_data['Month'] = input_data['Month'].astype(int)  # Ensure Month is an integer
        except KeyError as e:
            st.error(f"Column not found in uploaded CSV: {e}")
        except Exception as e:
            st.error(f"Error during encoding: {e}")

        if st.button("Predict Batch Data"):
            results = predict_batch(input_data)
            st.write("Prediction Results:")
            st.dataframe(results.head())

            # Provide option to download the prediction results as a CSV
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name='water_quality_predictions.csv',
                mime='text/csv'
            )






