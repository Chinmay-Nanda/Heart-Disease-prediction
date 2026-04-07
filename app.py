import streamlit as st
import pandas as pd
import joblib

model= joblib.load("LR_Heart.pkl")
scaler= joblib.load("scaler.pkl")
columns_expected=joblib.load("columns (1).pkl")
st.title("Heart stroke prediction by Chinu Bhai")
st.markdown("provide reqired details")
age=st.slider("Age",18,100,40)
sex=st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("chest_pain_type",["ATA","NAP","TA","ASY"])
Resting_bp= st.number_input("Resting blood press (mg Hg)",80,200,120)
Cholesterol= st.number_input("Input cholesterol (mg/dl)",100,600,200)
FastingBs= st.selectbox("Fasting Blood Sugar>120 mg/dl",[0,1])
ECG = st.selectbox("Resting Ecg",["Normal","ST","LVH"])
Max_HeartRate=st.slider("MaxHR",60,220,150)
Exercise_angina= st.selectbox("ExerciseAngina",["y","N"])
OldPeak= st.slider("OldPeak(Depression)",0.0,6.0,1.0)
ST_Slope= st.selectbox("Enter STslope",["up","Flat","Down"])

if st.button("Predict"):
    inputs={
        'Age':age,
        'RestingBP':Resting_bp,
        'Cholesterol':Cholesterol,
        'FastingBS': FastingBs,
        'MaxHr':Max_HeartRate,
        'OldPeak':OldPeak,
        'Sex' + sex: 1,
        'chest_pain_Type' + chest_pain:1,
        'Resting_ECG' + ECG: 1,
        'Exercise_angina' + Exercise_angina: 1,
        'ST_Slope' + ST_Slope:1    
        }
    
    Input_Df= pd.DataFrame([inputs])
    for col in columns_expected:
        if col not in Input_Df.columns:
            Input_Df[col]=0
    Input_Df= Input_Df[columns_expected]  
    Scaler_input= scaler.transform(Input_Df)      
    prediction= model.predict(Scaler_input)[0]

    if prediction==1:
        st.error("Risk Of Heart Disease")
    else:
        st.success("Low Chance")




    




 