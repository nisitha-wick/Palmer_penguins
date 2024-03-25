import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data is obtained form the famous palmer penguin dataset                  

""")

st.sidebar.header("User Input Features")

uploaded_file = st.sidebar.file_uploader("Upload your input csv file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_in_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_in_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_in_mm = st.sidebar.slider('Flipper length (mm)', 13.1, 21.5, 17.2)
        body_mass_in_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_in_mm': bill_length_in_mm,
                'bill_depth_in_mm': bill_depth_in_mm,
                'flipper_length_in_mm': flipper_length_in_mm,
                'body_mass_g': body_mass_in_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features


    input_df = user_input_features()

penguins_raw = pd.read_csv('C:\\Users\\wickn\\Desktop\\Palmer Penguins\\penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]

st.subheader('User Input Features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Waiting for the csv file to be uploaded. Currently using the example input parameters")
    st.write(df)

load_clf = pickle.load(open('penguins_classfier.pkl', 'rb'))

prediction = load_clf.predict(df)
prediction_probability = load_clf.predict_proba(df)

st.subheader("Prediction")
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probability)

