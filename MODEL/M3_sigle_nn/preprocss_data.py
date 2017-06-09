#coding=utf-8
from __future__ import division
import numpy as np
import pandas 
from evaluation import * 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)

model = Sequential()
model.add(Dense(30, activation='tanh', input_dim=40,kernel_initializer='random_uniform'))
model.add(Dense(1, activation='linear',kernel_initializer='random_uniform',bias_initializer='zeros'))
model.compile(loss=MAPE_loss,
          optimizer=adam,
          metrics=[MAPE_loss])