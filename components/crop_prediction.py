import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from components.scheme_recommendation import suggest_schemes  

# Load data
df = pd.read_csv('crop_recommendation.csv')
msp = pd.read_csv('msp_2024.csv')
num_col = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_col = 'label'
X = df[num_col]
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), num_col)]
)
model = RandomForestClassifier()
pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
pipe.fit(X_train, y_train)

def crop_prediction():
    st.title("Crop Prediction")
    st.header("Enter the following details:")

    N = st.number_input("Nitrogen content (N)", min_value=0, max_value=100, value=68)
    P = st.number_input("Phosphorus content (P)", min_value=0, max_value=100, value=58)
    K = st.number_input("Potassium content (K)", min_value=0, max_value=100, value=38)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=23.22)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=83.03)
    ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=6.3)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=221.20)

    if st.button("Predict Crop"):
        input_data = pd.DataFrame({
            'N': [N], 'P': [P], 'K': [K],
            'temperature': [temperature], 'humidity': [humidity],
            'ph': [ph], 'rainfall': [rainfall]
        })

        predicted_probabilities = pipe.predict_proba(input_data)[0]
        top_3_indices = predicted_probabilities.argsort()[-3:][::-1]
        top_3_crops = [pipe.classes_[index] for index in top_3_indices]

        st.write("Predicted Crops:")
        for i, crop in enumerate(top_3_crops, start=1):
            # Get suggested schemes for the crop
            schemes = suggest_schemes(crop)
            
            # Format schemes as bullet points
            scheme_list = ""
            if schemes:
                scheme_list = "<ul>" + "".join(f"<li>{scheme}</li>" for scheme in schemes) + "</ul>"
            else:
                scheme_list = "<p>No schemes found.</p>"

            msp_price = msp.loc[msp['Crops'].str.contains(crop, case=False, na=False), 'MSP 2024 (₹/quintal)']
            msp_price_text = f"<strong>MSP Price:</strong> ₹{msp_price.values[0]:,.2f} /quintal" if not msp_price.empty else "<strong>MSP Price:</strong> Not available"

            # Create the card using markdown
            card_html = f"""
            <div style="
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 10px;
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1);
            ">
                <h3 style="color: #5e0797;">{i}. {crop}</h3>
                {msp_price_text} <br/>
                <strong>Suggested Schemes:</strong>
                {scheme_list}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
