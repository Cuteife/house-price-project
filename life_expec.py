import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

# Load the saved model and scaler
with open(r"besr_rf_pipeline.pkl", "rb") as model_file:
    pipeline = pickle.load(model_file)
# pipeline = pickle.load('besr_rf_pipeline.pkl')

cities= ['Shoreline', 'Seattle', 'Kent', 'Bellevue', 'Redmond',
       'Maple Valley', 'North Bend', 'Lake Forest Park', 'Sammamish',
       'Auburn', 'Des Moines', 'Bothell', 'Federal Way', 'Kirkland',
       'Issaquah', 'Woodinville', 'Normandy Park', 'Fall City', 'Renton',
       'Carnation', 'Snoqualmie', 'Duvall', 'Burien', 'Covington',
       'Inglewood-Finn Hill', 'Kenmore', 'Newcastle', 'Mercer Island',
       'Black Diamond', 'Ravensdale', 'Clyde Hill', 'Algona', 'Skykomish',
       'Tukwila', 'Vashon', 'Yarrow Point', 'SeaTac', 'Medina',
       'Enumclaw', 'Snoqualmie Pass', 'Pacific', 'Beaux Arts Village',
       'Preston', 'Milton'
    ]

# Streamlit UI
st.title("Life Expectancy Prediction App")

# User input
city = st.selectbox("Select City", cities)
yr_built = st.slider("Year Built", 1800, 2027, 2011)
yr_renovated = st.slider("Year renovated", 1800, 2027, 2011)
bedrooms = st.number_input("Number of Bedrooms", min_value=0)
sqft_lot = st.number_input("sqft lot", min_value=0)
sqft_living = st.number_input("sqft_living", min_value = 0 )
floors = st.number_input("Floors")
waterfront = st.number_input("Waterfront (%)")
view = st.number_input("View")
condition = st.number_input("condition")
sqft_above = st.number_input("sqft above")
sqft_basement = st.number_input("sqft_basement")
bathrooms = st.number_input("bathrooms")



input_data = {
    'bedrooms' : [bedrooms], 
    'bathrooms' : [bathrooms], 
    'sqft_living' : [sqft_living], 
    'sqft_lot' : [sqft_lot],
    'floors' : [floors], 
    'waterfront' : [waterfront], 
    'view' : [view], 
    'condition' : [condition], 
    'sqft_above' : [sqft_above],
    'sqft_basement' : [sqft_basement], 
    'yr_built' : [yr_built], 
    'yr_renovated' : [yr_renovated], 
    'city' : [city]
}

input_data = pd.DataFrame(input_data)
# Predict
if st.button("Predict House Price"):
    prediction = pipeline.predict(input_data)
    st.success(f"Predicted House price: ${prediction[0]:.2f}")
    #okS