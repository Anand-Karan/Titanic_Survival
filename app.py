import streamlit as st
import pandas as pd
import numpy as np
from model import prep_data, load_model

# Loading model
model = load_model('logreg_model.pkl')

def user_input_features():
    Pclass = st.sidebar.selectbox('Pclass (Ticket Class)', options=[1, 2, 3])
    Sex = st.sidebar.selectbox('Sex', options=['male', 'female'])
    Age = st.sidebar.slider('Age', min_value=0, max_value=100, value=30)
    SibSp = st.sidebar.slider('SibSp (Number of siblings/spouses aboard)', min_value=0, max_value=10, value=0)
    Parch = st.sidebar.slider('Parch (Number of parents/children aboard)', min_value=0, max_value=10, value=0)
    Fare = st.sidebar.slider('Fare', min_value=0.0, max_value=100.0, value=50.0)
    Embarked = st.sidebar.selectbox('Embarked (Port of Embarkation)', options=['C', 'Q', 'S'])

    data = {
            'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': Embarked
            }

    features = pd.DataFrame(data, index=[0])
    return features

def main():
    st.write("""
    # Titanic Survivor Prediction App
    This app predicts whether a passenger on the Titanic would survive!
    """)
    
    df = user_input_features()
    prep_df = prep_data(df) # Pass user inputs through data preprocessing

    st.subheader('User Input parameters')
    st.write(df)

    prediction = model.predict(prep_df)
    st.subheader('Prediction')
    st.write("Survived" if prediction[0] == 1 else "Did Not Survive")

if __name__ == '__main__':
    main()
