#!/usr/bin/env python
# coding: utf-8

# In[26]:


import joblib
import streamlit as st
import random
import os

#loading model and count_vector
def load_svc():
    loaded_model = joblib.load("best_svc_model.pkl")
    return loaded_model

def load_count_vector():
    loaded_model = joblib.load("count_vectorizer.pkl")
    return loaded_model

st.title("Welcome to Text Sentiment Analysis")

st.write("Understand the emotions behind text with the click of a button. Analyze written content to gauge whether it's positive, negative, or neutral in sentiment.")




if 'counter' not in st.session_state:
    st.session_state.counter = 0

# Increment the counter whenever the app is rerun
st.session_state.counter += 1

# Display the counter in the top right corner
st.markdown(f'<div style="position: fixed; top: 10px; right: 10px; background-color: #e0e0e0; padding: 10px; border-radius: 5px;">Visits: {st.session_state.counter}</div>', unsafe_allow_html=True)



#function to select and display gif
def load_image(image_directory):
    image_list = [f for f in os.listdir(image_directory) if f.endswith((".gif"))]
    choice = random.choice(image_list)
    st.image(image_directory + choice)

# Load the model
svc_model = load_svc()
vect = load_count_vector()

# User input
user_input = st.text_input("Enter text: ")

if user_input:
    
    if user_input.lower() == 'oye papaji':
        st.image("gogi.jpg", caption="Optional image caption")
        st.write("**Balle Balle**")
        
    # Make predictions using the loaded model
    res = vect.transform([user_input])

    prediction = svc_model.predict(res)
    st.write("Sentiment of Text is :", "**" + str(prediction[0]) + "**")

    if prediction == 'Neutral':
        directory = "Neutral/"
        load_image(directory)
        

    if prediction == 'Positive':
        directory = "Positive/"
        load_image(directory)
        
    if prediction == 'Negative':
        directory = "Negative/"
        load_image(directory)
   
        


