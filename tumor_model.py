# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:18:04 2018

@author: Jack
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('data.csv')

#deleting unneeded rows
del df["Unnamed: 32"]
del df["id"]
df["diagnosis"].replace("M",1,inplace=True)
df["diagnosis"].replace("B",0,inplace=True)
df["Diagnosis"] = df["diagnosis"]
del df["diagnosis"]

#separating data into training, dev, and test sets
training_df = df.values[0:500]
dev_df = df.values[500:535]
test_df = df.values[535:]

train_x, train_y, test_x, test_y, dev_x, dev_y = [], [], [], [], [], []

#inserting data into training, dev, and test sets
for row in training_df:
    row = row.tolist()
    train_x.append(row[0:29])
    train_y.append(int(row[30]))
    
for row in dev_df:
    row = row.tolist()
    dev_x.append(row[0:29])
    dev_y.append(int(row[30]))

for row in test_df:
    row = row.tolist()
    test_x.append(row[0:29])
    test_y.append(int(row[30]))

#normalizing data
train_x, train_y = sklearn.preprocessing.normalize(np.array(train_x)), np.array(train_y)
dev_x, dev_y = sklearn.preprocessing.normalize(np.array(dev_x)), np.array(dev_y)
test_x, test_y = sklearn.preprocessing.normalize(np.array(test_x)), np.array(test_y)

#creating the model structure
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=29))
model.add(Dense(units=48, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#training the model
model.fit(train_x, train_y, epochs=5, batch_size=1, shuffle=True)
print("\n")

#using model to predict dev set
score = model.evaluate(dev_x, dev_y)
print("----Dev Set----\nLoss: " + str(score[0]) + "\nAccuracy: " + str(round(score[1]*100, 3))+"%\n")

#using model to predict test set
score = model.evaluate(test_x, test_y)
print("----Test Set----\nLoss: " + str(score[0]) + "\nAccuracy: " + str(round(score[1]*100, 3))+"%")

#separating data labels for graphing
malignant = sklearn.preprocessing.normalize(df[df["Diagnosis"]==1].values)
benign = sklearn.preprocessing.normalize(df[df["Diagnosis"]==0].values)

#using PCA to reduce dimensionality of the data to 2 features for graphing purposes
pca = PCA(n_components=2)
malignant = pca.fit_transform(malignant)
benign = pca.fit_transform(benign)

#using pyplot to graph data
for t in range(2):
    if t == 0:
        plt.scatter([pair[0] for pair in benign], [pair[1] for pair in benign], color='r', s=10, label="Benign")
    else:
        plt.scatter([pair[0] for pair in malignant], [pair[1] for pair in malignant], color='g', s=10, label="Malignant")
plt.legend()

