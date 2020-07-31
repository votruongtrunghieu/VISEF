import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import LabelEncoder, StandardScaler
dataset = pd.read_csv('data/breast-cancer-wisconsin.csv')
X = dataset.iloc[:,2:32]
y = dataset.iloc[:,1]
#Encoding M = 1, B = 0
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Features scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#
classifier = Sequential()
classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 100)
_, teAcc = classifier.evaluate(X_test, y_test)
_, trAcc = classifier.evaluate(X_train, y_train)
#classifier.save("breastCancerANN.h5")
