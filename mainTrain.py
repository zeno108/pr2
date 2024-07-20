import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

image_directory= 'dataset/'
no_tumor_image=os.listdir(image_directory+ 'Normal/')
tumor_image=os.listdir(image_directory+ 'Malignant/')
dataset=[]
label=[]
INPUT_SIZE=64
#print(no_tumor_image)
#path='2013_BC000541_ CC_L.jpg'
#print(path.split('.')[1])

for i , image_name in enumerate(no_tumor_image):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'Normal/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)
for i , image_name in enumerate(tumor_image):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+ 'Malignant/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

#print(dataset)
#print(len(label))

dataset=np.array(dataset)
label=np.array(label)
x_train, x_test, y_train, y_test= train_test_split(dataset, label, test_size=0.2, random_state=0)

#Reshape = (n, image_width, image_height, n_channel)

print(x_train.shape) #(1592, 64, 64, 3)


#print(x_train.shape) #(1592, 64, 64, 3)
#print(y_train.shape) #(1592,)
#print(x_test.shape) #(399, 64, 64, 3)
#print(y_test.shape) #(399,)

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)



# Model Building
# 64,64,3

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entryopy= 2 , softmax

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, 
batch_size=16,                                                  
verbose=1, epochs=10, 
validation_data=(x_test, y_test),
shuffle=False)


model.save('Breastcancer10Epochs.h5')

