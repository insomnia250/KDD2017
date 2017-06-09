#coding=utf-8
from __future__ import division
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

weather = pd.read_csv('../dataset/dataSets/training/weather (table 7)_training.csv')
weather['date'] = pd.to_datetime(weather['date'])
print weather.info()

newdf = pd.DataFrame()

daterange = pd.date_range(pd.datetime(2016,7,1),pd.datetime(2016,10,17),freq='D')

for hour in weather['hour'].unique():
	hourData = weather[weather['hour']==hour]
	newData = pd.DataFrame({'date':daterange})
	newData = pd.merge(newData, hourData,on='date',how='left')
	newData.fillna(method ='ffill',inplace=True)
	newdf = pd.concat([newdf,newData],axis=0,ignore_index=True)

newdf.sort_values(by=['date','hour']).to_csv('../dataset/dataSets/training/new_weather (table 7)_training.csv',index=False)