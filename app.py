import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('LinearRegressionModel.pkl','rb'))

st.header('CAR PRICE PREDICTION MODEL')

car = pd.read_csv('quikr_car - quikr_car.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()
car['name'] = car['name'].apply(get_brand_name)

name = st.selectbox('Select Car Name', car['name'].unique())
company = st.selectbox('Select Car Brand', car['company'].unique())
year = st.slider('Car Manufactured Year', 1994,2024)
kms_driven = st.slider('No of KMS DRIVEN', 1,200000)
fuel_type = st.selectbox('Fuel Type', car['fuel_type'].unique())

if st.button("PREDICT"):
    predict = pd.DataFrame(
    [[name,company,year,kms_driven,fuel_type]],
    columns=['name','company','year','kms_driven','fuel_type'])
    
    
    predict['fuel_type'].replace(['Diesel', 'Petrol', 'LPG'],[1,2,3], inplace=True)
    predict['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)
    
    
st.write(predict)
    

    
