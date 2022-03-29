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
    minp=0.0
    maxp=0.0
    min_price = np.round(min(linear_price, forest_price), 2)
    max_price = np.round(max(linear_price, forest_price), 2)
    col3,col4 = st.columns(2)
    with col3:
        if min_price >= 100:
            minp = np.abs(np.round(min_price/100,2))
            st.success('Minimum Prediction = Rs.   {} crores'.format(minp))
        else:
            minp = np.abs(np.round(min_price,2))
            st.success('Minimum Price Prediction = Rs.   {} lacs'.format(minp))
    with col4:
        if max_price >= 100:
            maxp = np.abs(np.round(max_price/100,2))
            st.success('Maximum Price Prediction = Rs. {} crores'.format(maxp))
        else:
            maxp = np.abs(np.round(max_price,2))
            st.success('Maximum Price Prediction = Rs. {} lacs'.format(maxp))


    average_price = (min_price + max_price) / 2
    if(min_price<100):
        min_price_sqft = (min_price / area_input) * 100000
    else:
        min_price_sqft = (min_price / area_input) * 10000
    if(max_price<100):
        max_price_sqft = (max_price/area_input)*100000
    else:
        max_price_sqft = (max_price / area_input) * 10000

    prices = {
        'Location': location_input,
        'BHK': int(bhk_input),
        'Bathrooms': int(bath_input),
        'Area (in sqft.)':np.round(area_input,2),
        'Minimum Price (in lacs) ':min_price,
        'Minimum Price per sqft.':min_price_sqft,
        'Maximum Price (in lacs) ':max_price,
        'Maximum Price per sqft.': max_price_sqft,
        'Average Price (in lacs)': average_price
    }
    df = pd.DataFrame(prices,index=[0])
    df2 = df.transpose()
    st.table(df)
