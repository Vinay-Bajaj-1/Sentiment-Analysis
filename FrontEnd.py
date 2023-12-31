#!/usr/bin/env python
# coding: utf-8

# In[26]:


import joblib
import streamlit as st
import random
import os

#loading model and count_vector
def load_svc():
    loaded_model = joblib.load("best_model.pkl")
    return loaded_model

def load_count_vector():
    loaded_model = joblib.load("count_vectorizer.pkl")
    return loaded_model

st.title("Welcome to Text Sentiment Analysis")

st.write("Understand the emotions behind text by writing in this text box. Analyze written content to gauge whether it's positive, negative, or neutral in sentiment.")



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
    
    if 'oye papaji'in user_input.lower():
        st.image("gogi.jpg", caption="Optional image caption")
        st.write("**Balle Balle**")
        st.audio('Audio/oye_papaji.mp3', format="audio/mp3")
        
    # Make predictions using the loaded model
    res = vect.transform([user_input])

    prediction = svc_model.predict(res)
    
    st.markdown(f"<p style='font-size: 16px;'>Sentiment of Text is: <b style='font-size: 20px;'>{prediction[0]}</b></p>", unsafe_allow_html=True)

    if prediction == 'Neutral':
        directory = "Neutral/"
        load_image(directory)
        

    if prediction == 'Positive':
        directory = "Positive/"
        load_image(directory)
        
    if prediction == 'Negative':
        directory = "Negative/"
        load_image(directory)
   
    


