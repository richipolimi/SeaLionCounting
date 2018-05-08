from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from keras.optimizers import SGD

# create the base pre-trained model
base_model = VGG16(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
#Do we really want this though?
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a hidden layer between our pretrained CNN and our ouptput layer
x = Dense(1024, activation='relu')(x)

# the output layer -- we have 2 classes (Sealion no sealion)
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#Our cnn is pretrained, we don't want to train it again (for now).
for layer in base_model.layers:
    layer.trainable = False

    
####THIS IS ONLY IF WE WANT TO FINE TUNE THE FINAL LAYERS OF THE CNN, HAVENT CHANGED ANYTHING FROM KERAS EXAMPLE
"""
# compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

"""

# SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='mean_squared_error')


#TRAIN, REPLACE X AND Y
model.fit_generator(x,y)
