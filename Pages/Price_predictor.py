import numpy as np
import streamlit as st
import pickle
import pandas as pd


st.set_page_config(page_title='Price Predictor')

with  open('df.pkl','rb') as file:
    df = pickle.load(file)

from collections import UserList
import sklearn.compose._column_transformer as ct

class _RemainderColsList(UserList):
    pass

ct._RemainderColsList = _RemainderColsList

import pickle
with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


st.header("Enter Your Inputs")

property_type = st.selectbox('Property Type',['flat','house'])
sector = st.selectbox('Sector',sorted(df.sector.unique().tolist()))
bedRoom = float(st.selectbox('No of Bedrooms',sorted(df.bedRoom.value_counts().index)))
bathroom = float(st.selectbox('No of Bathrooms',sorted(df.bathroom.value_counts().index)))
balcony = float(st.selectbox('No of Balconies',sorted(df.balcony.value_counts().index)))
facing = st.selectbox('Facing',df.facing.value_counts().index,placeholder='East')
agePossession = st.selectbox('Property Age',sorted(df.agePossession.value_counts().index))
Built_up_area = float(st.number_input('Built Up Area',placeholder=1000))
servant_room = st.selectbox('Servant Room',df['servant room'].value_counts().index)
store_room = st.selectbox('Store Room',df['store room'].value_counts().index)
furnishing_type = st.selectbox('Furnishing Type',df.furnishing_type.value_counts().index)
facilities = st.selectbox('Facilities',df.facilities.value_counts().index)
floor_category = st.selectbox('Floor Category',df.floor_category.value_counts().index)
if st.button('Predict'):
    # form a DataFrame
    columns = df.columns
    Data = [property_type, sector, bedRoom, bathroom, balcony, facing, agePossession, Built_up_area, servant_room, store_room, furnishing_type, facilities, floor_category]
    one_df = pd.DataFrame([Data],columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = round(base_price -0.25,2)
    high = round(base_price +0.25,2)
    #
    st.success("The Price of property may vary between ₹{}Cr to ₹{}Cr".format(low,high))