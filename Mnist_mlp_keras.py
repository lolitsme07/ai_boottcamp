import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Preprocess the data
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(x_train[0])
#plt.imshow(x_train[0])  # Print the first training image array
#plt.show()
print("************************")
print(f"label is :{y_train[0]}")

#normalise
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0   

#to_categorical

print(f"before:label is:{y_train[100]}")
y_train = to_categorical(y_train)
print(f"after:label is:{y_train[100]}")
y_test=to_categorical(y_test) 
model = Sequential()    

#architechture

model.add(Flatten(input_shape=(28, 28)))  
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compile
model.compile(optimizer='adam', loss='categorical_crossentropy')

#train
model.fit(x_train, y_train, epochs=10, batch_size=64, )

print("################################")
