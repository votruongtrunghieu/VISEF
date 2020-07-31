from numpy import loadtxt
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.layers import Dense
dataset = loadtxt('data/pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]
scaler = StandardScaler()
X = scaler.fit_transform(X)
model = Sequential()
model.add(Dense(units=1000, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(units=1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=10)
#_, accuracy = model.evaluate(X, y)
model.save("diabetes.h5")