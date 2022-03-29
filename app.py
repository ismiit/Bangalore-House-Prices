import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import time as time

st.title('Bangalore House Prices Predictor')
locations = pickle.load(open('locations.pkl', 'rb'))

cola,colb = st.columns(2)
with cola:
    image = Image.open('images/bang.jpg')
    image = image.resize((500,224))
    st.image(image)
with colb:
    image2 = Image.open('images/house.jpg')
    image2 = image2.resize((500, 224))
    st.image(image2)


location_input = st.selectbox('Select Location', locations)

col1,col2 = st.columns(2)
with col1:
    bhk_input = st.number_input('BHK SIZE')
with col2:
    bath_input = st.number_input('BATHROOMS')

area_input = st.number_input('AREA expected (in sqft.)')

linear = pickle.load(open('lr_model.pkl', 'rb'))
forest = pickle.load(open('rf_model.pkl', 'rb'))
ridge = pickle.load(open('rid_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

def predict_price(model,location,sqft,bath,bhk):
    loc_index = np.where(columns==location)[0][0]

    x = np.zeros(len(columns))
    x[0] = bath
    x[1] = bhk
    x[2] = sqft
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]

if st.button('Predict'):
    with st.spinner('Predicting'):
        time.sleep(2)
    linear_price = predict_price(linear, location_input, area_input, bath_input, bhk_input)
    forest_price = predict_price(forest, location_input, area_input, bath_input, bhk_input)
    ridge_price = predict_price(ridge, location_input, area_input, bath_input, bhk_input)
    lp=0.0
    rp=0.0
    col3,col4 = st.columns(2)
    with col3:
        if linear_price >= 100:
            lp = np.abs(np.round(linear_price/100,2))
            st.success('Linear Model Prediction = Rs. {} crores'.format(lp))
        else:
            lp = np.abs(np.round(linear_price,2))
            st.success('Linear Model Prediction = Rs. {} lacs'.format(lp))
    with col4:
        if forest_price >= 100:
            rp = np.abs(np.round(forest_price/100,2))
            st.success('Ridge Model Prediction = Rs. {} crores'.format(rp))
        else:
            rp = np.abs(np.round(forest_price,2))
            st.success('Ridge Model Prediction = Rs. {} lacs'.format(rp))
    prices = {
        'Location': location_input,
        'BHK': int(bhk_input),
        'Bathrooms': int(bath_input),
        'Area (in sqft.)':np.round(area_input,2),
        'Minimum Price (Rs.)':np.round(min(lp,rp),2),
        'Maximum Price (Rs.)':np.round(max(lp,rp),2),
    }
    df = pd.DataFrame(prices,index=[0])
    st.dataframe(df)