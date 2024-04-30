import streamlit as st
import numpy as np
from predictor import do_prediction
import pandas as pd

Xtrainall = pd.read_csv("Training.csv")
disease = Xtrainall.iloc[:, -1]
disease = np.unique(disease)

labels = Xtrainall.columns[:-1]

if "None" not in labels:
    labels = np.insert(labels, 0, "None")


st.markdown("""
    <style>
        .title {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #0e1117;
            margin:25px 0;
            padding: 10px 0;
            text-align: center;
            color: white;
            z-index: 1000;
        }
        .pred {
            text-align:center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title' style='text-align: center; color: white;'><font size=50>SymptoSCAN</font></h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Enter your top 5 symptoms <font size=2>(max.)</font></h4>", unsafe_allow_html=True)

selected_symptoms = []
for i in range(5):
    options = [symptom for symptom in labels if symptom not in selected_symptoms]
    
    selected_symptom = st.selectbox(f'Symptom {i+1}', options,index=0)

    if selected_symptom != 'None':
        selected_symptoms.append(selected_symptom)
    

predict = st.button("predict")

if predict: 
    results = do_prediction(selected_symptoms)
    for i in results:
        st.markdown(f"<h3 class='pred'><u>{i}</u></h3>", unsafe_allow_html=True)


