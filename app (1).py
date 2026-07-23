import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import streamlit as st
import pandas as pd
import pickle

with open('ford_model.pkl', 'rb') as f:
    model = pickle.load(f, encoding='latin1')

with open('le_model.pkl', 'rb') as f:
    le_model = pickle.load(f, encoding='latin1')

with open('le_trans.pkl', 'rb') as f:
    le_trans = pickle.load(f, encoding='latin1')

st.title("Ford Car Price Predictor")

model_name = st.selectbox("Model", le_model.classes_)
year = st.number_input("Year", 2010, 2025, 2018)
trans = st.selectbox("Transmission", le_trans.classes_)
fuel = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
mileage = st.number_input("Mileage", 0, 200000, 50000)
engine = st.number_input("Engine Size", 1.0, 5.0, 2.0)
mpg = st.number_input("MPG", 10, 100, 40)

if st.button("Predict Price"):
    model_enc = le_model.transform([model_name])[0]
    trans_enc = le_trans.transform([trans])[0]
    input_data = pd.DataFrame([[model_enc, year, trans_enc, fuel, mileage, engine, mpg]], 
                              columns=['model','year','trans','fuel','mileage','engine','mpg'])
    price = model.predict(input_data)[0]
    st.success(f"Predicted Price: £{price:,.0f}")
