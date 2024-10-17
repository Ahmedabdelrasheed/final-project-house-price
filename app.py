import streamlit as st
from tensorflow.keras.models import load_model
import pickle as pl
import pandas as pd
import numpy as np

# Load model and preprocessors
model = load_model("model/model.keras")
sc = pl.load(open("model/scaler.pkl", 'rb'))
encoder = pl.load(open("model/encoder.pkl", 'rb'))

st.title("NY HOUSE PRICING")

# Collecting inputs
house_type = st.selectbox("house_type", ('Condo for sale', 'House for sale', 'Townhouse for sale', 'Co-op for sale', 'Multi-family home for sale'))
house_sublocality = st.selectbox("house_sublocality", ('Manhattan', 'New York County', 'Richmond County', 'Kings County', 'New York', 'East Bronx', 'Brooklyn', 'The Bronx', 'Queens', 'Staten Island', 'Queens County', 'Bronx County', 'Coney Island', 'Brooklyn Heights', 'Jackson Heights', 'Riverdale', 'Rego Park', 'Fort Hamilton', 'Flushing', 'Dumbo', 'Snyder Avenue'))
house_bath = st.number_input('Number of baths', min_value=0, step=1, format="%d")
house_bed = st.number_input('Number of beds', min_value=0, step=1, format="%d")
house_sqft = st.number_input('Square footage', min_value=0, step=1, format="%d")

# Creating the DataFrame
df = pd.DataFrame([[house_type, house_bed, house_bath, house_sqft, house_sublocality]], columns=["TYPE", "BEDS", "BATH", "PROPERTYSQFT", "SUBLOCALITY"])

df_numeric =["BEDS", "BATH", "PROPERTYSQFT"]

# Encode and scale the features
df_encoded = encoder.transform(df)
df_encoded[df_numeric] = sc.transform(df_encoded[df_numeric])

# Combine encoded categorical and scaled numerical data


# Make predictions
prediction = model.predict(df_encoded)
if st.button("Predict"):
  st.write("Predictions:",np.exp(prediction))
# Display the results
  
prediction_mean = np.mean(prediction)
prediction_std = np.std(prediction)
st.markdown(f"**price with Uncertainty**: {13.265204} ± {0.85088575}")


