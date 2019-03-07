from keras import layers
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

model = Sequential()
model.add(Dense(256, activation='sigmoid', input_dim=100))
model.add((Dropout(0.8)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

data = np.random.random((1000,100))
labels = np.random.randint(2, size=(1000,1))
model.fit(data, labels, epochs=1000)