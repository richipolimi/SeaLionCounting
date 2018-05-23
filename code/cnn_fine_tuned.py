from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten, Dropout
from keras import regularizers
from keras.optimizers import SGD
from keras.utils import np_utils
import keras
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from keras.preprocessing.image import ImageDataGenerator
#Hyper Paratmeters
batch_size = 256
epochs = 10
keras.backend.set_image_data_format("channels_last")
np.random.seed(123)
##Shuffle X,Y in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


"""
X = np.load('X.npy')
Y = genfromtxt('DS1.csv', delimiter =',').astype(np.float64)
Y = Y[:,1]
"""

X = np.load('X_final.npy')
Y = genfromtxt('DS3.csv', delimiter =',').astype(np.float64)
Y = Y[:,1]


#One off representation
#Y = np_utils.to_categorical(Y,num_classes=2)
print(Y.shape)
print(X.shape)
X,Y = unison_shuffled_copies(X,Y)

#normalize
X=X/255
#reshape, so that pretrained CNN can use this
X = X.reshape(X.shape[0],50,50,3)


#Split data into training,validation and testing
X_train = X[:40000,:]
Y_train = Y[:40000]

X_valid = X[40000:50000,:]
Y_valid = Y[40000:50000]

X_test = X[50000:,:]
Y_test = Y[50000:]


"""
#Split data into training,validation and testing
X_train = X[:25000,:]
Y_train = Y[:25000]

X_valid = X[25000:,:]
Y_valid = Y[25000:]

"""


#Check if we are using GPU or CPU 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#Load the base pre-trained model
#include-top removes the top 3 layers of VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape = (50,50,3))

# add a global spatial average pooling layer
#Do we really want this though? Because the VGG16 already has a pooling layer before this one.
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
# add a hidden layer between our pretrained CNN and our ouptput layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# the output layer -- we have 2 classes (Sealion no sealion)
predictions = (Dense(1, activation='sigmoid'))(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#Our cnn is pretrained, we don't want to train it again (for now).
for layer in model.layers[:10]:
    layer.trainable = False

for i, layer in enumerate(model.layers):
    print(i, layer.name)
        
"""
####THIS IS ONLY IF WE WANT TO PRETRAIN THE TOP LAYERS AND THEN FINE TUNE THE ALL LAYERS OF THE CNN,

# compile
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9 , nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])


# train the model on the new data for a few epochs, to make sure our classifier is well trained and we can tune top ~x layers accordingly
model.fit(x=X_train, y= Y_train, batch_size=256,
                    epochs=10,
                
                    validation_data=(X_test, Y_test))

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(model.layers):
   print(i, layer.name)

# we lets try fintuning the entire CNN
for layer in model.layers[:6]:
   layer.trainable = True
for layer in model.layers[6:]:
   layer.trainable = True

########################################### 
"""   
   
modelnew = model
besttest = 0
minimum = 0.0001
maximum = 0.01

bestlr = 0
bestlambda = 0
bestaccuracy = 0

minimum_lambda = 0
maximum_lambda = 0.0001
bestiteration = 0

for i in range(50):
    
    lr = minimum + (maximum-minimum) * np.random.rand() 
    lambda_reg = minimum_lambda + (maximum_lambda-minimum_lambda)*np.random.rand()
    print(str(i) + ", Learning rate: " + str(lr) ,  file=open("results.txt", "a"))
    print(str(i) + ", Lambda for regularization: "+ str(lambda_reg),  file=open("results.txt", "a"))
    
    weights = model.get_weights()
    modelold = keras.models.clone_model(model)
    modelold.set_weights(weights)
    model.layers[20].W_regularizer = regularizers.l2(lambda_reg)
    model.layers[22].W_regularizer = regularizers.l2(lambda_reg)
    sgd = SGD(lr, decay=1e-6, momentum=0.9 , nesterov=True)
    modelold.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    
    #TRAIN, REPLACE X AND Y
    #model.fit_generator(X_train,Y_train)
    
    history = modelold.fit(x=X_train,y=Y_train, batch_size=256,
                        epochs=epochs,
                        validation_data=((X_valid, Y_valid)))
    
    score = modelold.evaluate(X_valid, Y_valid, verbose=0)
    print(score)
    print('Validation loss:', score[0])
    print(str(i) + ', Validation accuracy:' + str( score[1]) + '\n',file=open("results.txt", "a") )
    if float(score[1])>besttest:
        besttest = float(score[1])
        bestlr  = lr
        bestlambda = lambda_reg    
        modelold.save('sealion_model_2_newdata.h5')
        bestiteration = i
print("Iteration " + str(bestiteration) +  " had best lambda at: " + str(bestlambda), file=open("results.txt", "a"))
print("Iteration " + str(bestiteration) +  " had best lr at: " + str(bestlr), file=open("results.txt", "a"))
print("Iteration " + str(bestiteration) +  " was best validation accuracy at: " + str(besttest), file=open("results.txt", "a"))

