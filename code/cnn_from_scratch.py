from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,MaxPooling2D,Conv2D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import keras
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#Hyper Paratmeters
np.random.seed(123)
batch_size = 256
epochs = 10
width= 50
keras.backend.set_image_data_format("channels_last")
##Shuffle X,Y in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


X = np.load('X_final.npy')
Y = genfromtxt('DS3.csv', delimiter =',').astype(np.float64)
Y = Y[:,1]
#One off representation#
#Y = np_utils.to_categorical(Y)
print(Y)
X,Y = unison_shuffled_copies(X,Y)

#normalize
X=X/255
#reshape, so that pretrained CNN can use this
X = X.reshape(X.shape[0],50,50,3)


#Split data into training and valid, and testing
X_train = X[:40000,:]
Y_train = Y[:40000]

X_valid = X[40000:50000,:]
Y_valid = Y[40000:50000]

X_test = X[50000:,:]
Y_test = Y[50000:]

#Check if we are using GPU or CPU 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(width,width,3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(0.001, decay=1e-6, momentum=0.9 , nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    


#TRAIN, REPLACE X AND Y
#model.fit_generator(X_train,Y_train)
history = model.fit(x=X_train,y=Y_train, batch_size=256,
                    epochs=epochs,
                    validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('sealion_model_4_newdata.h5')


