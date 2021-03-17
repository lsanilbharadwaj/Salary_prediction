# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:48:56 2021

@author: lsani
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:49:28 2021

@author: lsani
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
#from PIL import Image
#from flask import Flask, request, jsonify, render_template


#app = Flask(__name__)
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)



#@app.route('/')
def welcome():
    return "Welcome!!! Let's predict the salary"


def salary_prediction(experience,test_score,interview_score):
    salary = classifier.predict([[experience,test_score,interview_score]])
    print(salary)
    return salary
#@app.route('/predict',methods=['GET'])
def main():
    st.title("Salary Prediction")
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">Salary Prediction ML App </h2>
    </div> 
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    experience = st.text_input("experience","Type Here")
    test_score = st.text_input("test_score","Type Here")
    interview_score = st.text_input("interview_score","Type Here")
    result=""
    if st.button("Predict"):
        result=salary_prediction(int(experience),int(test_score),int(interview_score))
    st.success("The salary is {}".format(result))
    if st.button("About"):
        st.text("Predicted by Anil Bharadwaj")
    
    
    



if __name__ == '__main__':
    main()