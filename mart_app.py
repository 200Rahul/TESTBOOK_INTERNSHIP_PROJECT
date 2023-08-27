#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn import *
import pickle


# In[2]:


# Load the trained model
model = pickle.load(open('mart_data.pkl', 'rb'))
best_model = pickle.load(open('Mart_sales.pkl', 'rb'))
# Create a Streamlit app
st.title('Sales Prediction App')
# Input fields for user to provide feature values
customer_name = st.text_input('Customer Name')
city = st.text_input('City')
state = st.text_input('State')
product_name = st.text_input('Product Name')
actual_discount = st.slider('Actual Discount', min_value=0.0, max_value=100.0)
quantity = st.slider('Quantity', min_value=0, max_value=100)
order_day_of_week = st.slider('Order Day of Week', min_value=0, max_value=6)
order_day_of_year = st.slider('Order Day of Year', min_value=1, max_value=366)
order_year = st.slider('Order Year', min_value=2009, max_value=2023)
ship_day_of_week = st.slider('Ship Day of Week', min_value=0, max_value=6)
ship_day_of_year = st.slider('Ship Day of Year', min_value=1, max_value=366)
ship_year = st.slider('Ship Year', min_value=2009, max_value=2023)
days_to_ship = st.slider('Days to Ship', min_value=0, max_value=30)
profit = st.slider('Profit', min_value=0.0, max_value=1000.0)


# When a button is clicked, make predictions
if st.button('Predict Sales'):
    # Create a DataFrame from user inputs
    user_input = pd.DataFrame({
        'Customer Name': [customer_name],
        'City': [city],
        'State': [state],
        'Product Name': [product_name],
        'Actual Discount': [actual_discount],
        'Quantity': [quantity],
        'Order Day of Week': [order_day_of_week],
        'Order Day of Year': [order_day_of_year],
        'Order Year': [order_year],
        'Ship Day of Week': [ship_day_of_week],
        'Ship Day of Year': [ship_day_of_year],
        'Ship Year': [ship_year],
        'Days to Ship': [days_to_ship],
        'Profit': [profit]
    })

    # Make predictions
    prediction = model.predict(user_input)

    # Display the prediction
    st.write(f'Predicted Sales: {prediction[0]:.2f}')

