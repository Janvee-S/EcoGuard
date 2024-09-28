import streamlit as st
import pickle
import numpy as np

# Function to load the saved model and scaler
def load_model():
    with open('green_zone_classifier.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Load the model and scaler
model, scaler = load_model()

# Streamlit app interface
def show_predict_page():
    st.title("Green Zone Prediction")

    st.write("""### Input air quality data to predict if the area is a Green Zone or not.""")

    # Input fields for air quality features
    so2 = st.number_input("SO2 (Sulfur Dioxide)", min_value=0.0, value=5.0)
    no2 = st.number_input("NO2 (Nitrogen Dioxide)", min_value=0.0, value=10.0)
    rspm = st.number_input("RSPM (Respirable Suspended Particulate Matter)", min_value=0.0, value=80.0)
    air_quality_index = st.number_input("Air Quality Index", min_value=0, value=50)

    # Button to trigger prediction
    ok = st.button("Predict Green Zone")
    if ok:
        # Prepare the input data in the form of a numpy array
        X = np.array([[so2, no2, rspm, air_quality_index]])
        
        # Scale the input data using the saved scaler
        X_scaled = scaler.transform(X)
        
        # Use the trained model to make predictions
        green_zone_prediction = model.predict(X_scaled)
        
        # Output the prediction result
        if green_zone_prediction[0] == 1:
            st.subheader("This area is likely a Green Zone 🌳")
        else:
            st.subheader("This area is NOT a Green Zone 🚫")

# Call the function to display the prediction page
show_predict_page()
