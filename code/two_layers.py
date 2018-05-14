import numpy as np
from config import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import  matplotlib.pyplot as plt

def genrate_small_ds(X, Y, n=10000, ds_path='/'):
    rnd_indices = np.random.permutation(n)
    X1 = X[0:n][rnd_indices]
    Y1 = Y[0:n][rnd_indices]
    np.save(ds_path + 'X1', X1)
    np.save(ds_path + 'Y1', Y1)

def store_network(model, file_path):
    model.save(file_path)

def load_network(file_path):
    return load_model(file_path)

# load the dataset
#dataset_path = DATASET_DIR + 'DS/'
#X_file = dataset_path + 'X.npy'
#Y_file = dataset_path + 'DS.csv'

#X = np.load(X_file).astype(np.float)/255
#Y = np.loadtxt(Y_file, delimiter=',')[:, 1]

#genrate_small_ds(X, Y, n=6000, ds_path=dataset_path)


# load small dataset
dataset_path = DATASET_DIR + 'DS/'
X_file = dataset_path + 'X1.npy'
Y_file = dataset_path + 'Y1.npy'

X = np.load(X_file)
Y = np.load(Y_file)

# train val splitting
X_train, Y_train = X[0:6000], Y[0:6000]
X_val, Y_val = X[6000:10000], Y[6000:10000]

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=7500))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X, Y, validation_split=0.33, epochs=2, batch_size=128)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Train and Val Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#store_network(model, '/home/riccardo/git/SeaLionCounting/Model/2layers')
