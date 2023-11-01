#!/usr/bin/env python
# coding: utf-8

# In[26]:


import joblib
import streamlit as st

#loading model and count_vector
def load_svc():
    loaded_model = joblib.load("best_svc_model.pkl")
    return loaded_model

def load_count_vector():
    loaded_model = joblib.load("count_vectorizer.pkl")
    return loaded_model

st.title("Welcome to Text Sentiment Analysis")

st.write("Understand the emotions behind text with the click of a button. Analyze written content to gauge whether it's positive, negative, or neutral in sentiment.")

# Load the model
svc_model = load_svc()
vect = load_count_vector()

# User input
user_input = st.text_input("Enter text: ")

if user_input:
    
    if user_input.lower() == 'oye papaji':
        st.image("gogi.jpg", caption="Optional image caption")
        st.write("**Extremly Happy**")
        
    # Make predictions using the loaded model
    res = vect.transform([user_input])

    prediction = svc_model.predict(res)
    st.write("Sentiment of Text is :", "**" + str(prediction[0]) + "**")

    if prediction == 'Neutral':
        st.image("straight.jpg", caption="Optional image caption", use_column_width = False )

    if prediction == 'Positive':
        st.image("smile.png", caption="Optional image caption")
        
    if prediction == 'Negative':
        st.image("angry.png", caption="Optional image caption")
        
    if user_input.str.lower() == 'oye papaji':
        st.image("gogi.jpg", caption="Optional image caption")


