# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:38:08 2021

@author: lsani
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv('hiring.csv')


df.experience.fillna('zero',inplace=True)

median_test_score = df['test_score'].median()
df['test_score'].fillna(median_test_score,inplace=True)


df.experience = df.experience.apply(w2n.word_to_num) 

    
    
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df['salary'])


pickle.dump(reg,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[2,9,6]]))