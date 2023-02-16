from datetime import datetime
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import streamlit as st
import numpy as np
import glob
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

from datetime import date
from st_btn_select import st_btn_select

selection = st_btn_select(('CHECK YOUR PLANTS', 'PLANT DISEASES INFO', 'ABOUT PLANTDOC', 'CONTACT'))



    

if selection == 'CHECK YOUR PLANTS':
  
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"PNG"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('plantbackground.jpg')    
      

                              
    st.markdown(""" <style> .font {
    font-size:50px ; font-weight: 800; color: #2e0a06; background-color: #ff958a;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"> PlantDoc</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font2 {
    font-size:20px; color: ##0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">A Novel Plant Disease Classification Machine Learning Web Application</p>', unsafe_allow_html=True)
    

      
    

    st.markdown(""" <style> .font3 {
    font-size:35px ; font-weight: 600; color: #ff958a; background-color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">Detect if your plants have a disease or not via AI technology: receive a quick & convenient result within seconds!</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font5 {
    font-size:25px ; font-weight: 600; color: #2e0a06; background-color: #fcf6f5;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font5">Upload Plant Image Here</p>', unsafe_allow_html=True)
    
    image = st.file_uploader(label = " ", type = ['png','jfif', 'jpg', 'jpeg', 'tif', 'tiff', 'raw', 'webp'])
 
    def import_and_predict(image_data, model):
        size = (256, 256)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        #image = np.array(Image.open(image_data).resize((256, 256)))
        img = tf.keras.utils.img_to_array(image)
        img = tf.expand_dims(img, 0)
        probs = model.predict(img)
        score = tf.nn.softmax(probs[0])
        text = ("PlantDoc predicts that this is an image of a/an **{}**."
        .format(class_names[np.argmax(score)], 100 * np.max(score)))
        return text

  
    loaded_model = tf.keras.models.load_model('PlantDocModel2.h5', compile=False)
    #loaded_model = tf.keras.models.load_model('PlantDocModel2.h5')
    class_names = [
    'Apple with Black rot',
    'Apple that is healthy',
    'Cherry that is healthy',
    'Cherry with Powdery mildew',
    'Corn that is healthy',
    'Corn with Northern Leaf Blight',
    'Grape with Black rot',
    'Grape that is healthy',
    'Potato with Early blight',
    'Potato that is healthy',

    'Tomato with a Bacterial spot',
    
    'Tomato that is healthy',
    'Tomato with the Tomato mosaic virus'
    ]


    predictionText = "Prediction: Waiting for an image upload"

    if image is not None:
        st.image(image)
        predictionText = (import_and_predict(Image.open(image), loaded_model))

    st.markdown(predictionText)   
    #st.markdown('<p class="font2">predictionText</p>', unsafe_allow_html=True)
    
if selection == 'ABOUT PLANTDOC':
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"PNG"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('plantbackground.jpg')    
    
    st.markdown(""" <style> .font {
    font-size:50px ; font-weight: 800; color: #7792E3; background-color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About PlantDoc</p>', unsafe_allow_html=True)
   

   
  
    
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">Mission</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ;  color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">Due to the high usage of pesticides, many plant diseases have become resilient and more common. In addition, farmers, especially those living in isolated or rural parts of the world, may not know what or if a disease has affected their plant. Therefore it is essential to figure out what diseases are affecting plants so consumers do not get sick. So, the goal of **PlantDoc** is to provide the farmers and gardeners an opportunity to check the conditions of each and every plant they tend to. **PlantDoc** aims to make this checking process simpler and more convenient by utilizing AI & machine learning to self-diagnose affected plants. In addition, I want to help raise awareness for plant preservation and reducing plant diseases and build this app as a convenient platform with resources to educate every planter about their impacts on the earth and help them understand what choices they can make to raise healthier plants.
                
                
                
                </p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">How PlantDoc was Built</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">PlantDoc has two parts: the AI model and web app. The AI model is built using the TensorFlow framework in the Python Language while the web app is built using Streamlit using HTMl/CSS formatting. I trained the model in Google Colab on a dataset consisting of 13 types of plant conditions sourced from the Kaggle New Plant Diseases Dataset and deployed the model into this web app with Streamlit.</p>', unsafe_allow_html=True)
    
    st.image("DataFlowDiagramPlantDoc.jpg", caption='Diagram Detailing the Process of PlantDoc, from data input to web app deployment.')
                
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">Benefits and Potential Impacts of PlantDoc</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">PlantDoc is one of the first ever AI web apps to help diagnose plants. Diagnosing plants in their earliest stages of a disease leads to early treatment, healthier plants, and therefore healthier food for people and animals.</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">Future of PlantDoc</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">The accuracy of this CNN model is currently the highest at 94%, but I plan to improve the accuracy of the AI model even more. I also plan to partner with agricultural businesses so we can test out the app with farmers and gardeners in real-time.</p>', unsafe_allow_html=True)
    
   
    
    
    
    
if selection == 'PLANT DISEASES INFO':
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"PNG"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('plantbackground.jpg')    
    
    st.markdown(""" <style> .font {
    font-size:50px ; font-weight: 800; color: #7792E3; background-color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">About Common Plant Disease + How to Prevent Them</p>', unsafe_allow_html=True)
   

    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">About the Creators</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">The information page states relevant information for gardeners and farmers about the details of each disease and tips on how to prevent disease when growing plants. </p>', unsafe_allow_html=True)
  
    
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">Mission</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ;  color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">Due to the high usage of pesticides, many plant diseases have become resilient and more common. In addition, farmers, especially those living in isolated or rural parts of the world, may not know what or if a disease has affected their plant. Therefore it is essential to figure out what diseases are affecting plants so consumers do not get sick. So the goal of PlantDoc is to provide the farmers and gardeners an opportunity to check the conditions of each and every plant they tend to. PlantDoc aims to make this checking process simpler and more convenient by utilizing AI & machine learning.</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">How PlantDoc was Built</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">PlantDoc has two parts: the AI model and web app. The AI model is built using the TensorFlow framework in the Python Language while the web app is built using Streamlit using HTMl/CSS formatting. We trained the model in Google Colab on a dataset consisting of 15 types of plant conditions sourced from the PlantVillage dataset on Kaggle and deployed the model into the web app with Streamlit.</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">Future of PlantDoc</p>', unsafe_allow_html=True)

    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">The accuracy of our CNN model is currently 92%, but we plan to improve the accuracy of the AI model even more. We also plan to partner with agricultural businesses so we can test out the app with farmers.</p>', unsafe_allow_html=True)
    
    
       
if selection == 'CONTACT':
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"PNG"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    add_bg_from_local('plantbackground.jpg')    
    
    st.markdown(""" <style> .font {
    font-size:50px ; font-weight: 800; color: #7792E3; background-color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Contact PlantDoc Creator</p>', unsafe_allow_html=True)
   



    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #fffafa;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3"> Have a question? Email me for questions, website bugs, or concerns.</p>', unsafe_allow_html=True)
  
    if st.button('Email the Creator'):

        st.write("[Send me an email](mailto:justinh45700@gmail.com)")
    
    st.markdown(""" <style> .font2 {
    font-size:30px ; font-weight: 600; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font2">More Info</p>', unsafe_allow_html=True)
    
    st.markdown(""" <style> .font3 {
    font-size:20px ; color: #0a0302;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font3">PlantDoc Established since Nov 2022.</p>', unsafe_allow_html=True)

    
  
    st.markdown('<p class="font3"> Copyright © 2022 United States</p>', unsafe_allow_html=True)
