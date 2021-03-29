#import libraries
import numpy as np
import streamlit as st
from datetime import date
import pickle
import pandas as pd
import xgboost as xgb

#Load the model
model = xgb.XGBRegressor()
model.load_model('xg_final.model')

st.write("""
# Predicting Used Car Prices
This app predicts the ** used car prices ** for:
- PARF cars, < 10yo only
- excludes OPC cars
- excludes imported used cars 
\nand its **depreciation** using features input via the **side panel** 
""")
# Load the dataframe skeleton for prediction
df_skeleton = pd.read_csv('df_skeleton_1_4.csv', index_col = 0)
# Load the brand_list
brand_list = pickle.load(open('brand_list_v4.pkl', 'rb'))
# Load the modelsubmodel_db list
modelsubmodel_db = pickle.load(open('modelsubmodel_db_v4.pkl', 'rb'))


def addYears(d, years):
    try:
    # Return same day of the current year
        return d.replace(year=d.year + years)
    except ValueError:
    # If not same day, it will return other, i.e.  February 29 to March 1 etc.
        return d + (date(d.year + years, 1, 1) - date(d.year, 1, 1))


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox
    return type : pandas dataframe

    """
    make = st.sidebar.selectbox("Select Make", options = brand_list)
    model_submodel = st.sidebar.selectbox("Select Model-submodel", options= modelsubmodel_db[make])
    no_of_owners = st.sidebar.number_input('Number of Owners', min_value= 1)
    mileage = st.sidebar.number_input('Mileage(km)', min_value= 10)
    reg_date = st.sidebar.date_input('Car Registration Date', max_value= date.today())
    coe_qp = st.sidebar.number_input('COE QP ($)', min_value= 10000)
    arf = st.sidebar.number_input('ARF ($)', min_value = 100)

    coe_days_left = float((addYears(reg_date, 10) - date.today()).days -1)

    df_skeleton.loc[0, 'ARF'] = arf
    df_skeleton.loc[0, 'COE_LISTED'] = coe_qp
    df_skeleton.loc[0, 'DAYS_OF_COE_LEFT'] = coe_days_left
    df_skeleton.loc[0, 'NO_OF_OWNERS'] = no_of_owners
    df_skeleton.loc[0, 'MILEAGE_KM'] = mileage
    df_skeleton.loc[0, make] = 1
    df_skeleton.loc[0, model_submodel] = 1

    return df_skeleton, make, model_submodel, arf, coe_days_left

df_skeleton, make, model_submodel, arf, coe_days_left = get_user_input()
df_skeleton.fillna(value = 0, inplace = True)

st.subheader('Model input parameters(transformed)')
st.write(df_skeleton[[make, model_submodel, 'NO_OF_OWNERS', 'MILEAGE_KM', 'DAYS_OF_COE_LEFT', 'COE_LISTED', 'ARF']])


# when 'Predict' is clicked, make the prediction and store it
if st.sidebar.button("Predict"):
 result = int(np.exp(model.predict(df_skeleton.values)[0]))
 st.success('Estimated pricing of vehicle is : ${:,}'.format(result))
 parf = 0.5 * arf
 depreciation = int((result - parf) / (coe_days_left / 365))
 st.success('Estimated depreciation is : ${:,} /year'.format(depreciation))



