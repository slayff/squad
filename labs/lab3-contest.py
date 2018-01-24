
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import skimage
import sklearn
from tqdm import tqdm
import os
import gc
import h5py as h5py

from skimage.transform import resize
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.layers.noise import AlphaDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

from IPython.display import clear_output


# In[2]:


DATA_PATH = './data/'
DATA_FILES = ['train-1.npy', 'train-2.npy', 'train-3.npy', 'train-4.npy']
OUT_PATH = './data_binary/'


# In[ ]:


def preprocess_data(filenames, fixed_labels_dict={}, img_size=68, verbose=0, thresholding=0):
    data = []
    labels = []
    ind = 0
    for filename in filenames:
        if verbose:
            print('Processing file %s' %filename)
        full_filename = DATA_PATH + filename
        tmp_data = np.load(full_filename)
        
        for entry in tqdm(tmp_data):
            img, label = entry
            resized_img = resize(img, (img_size, img_size), mode='constant')
            if (thresholding):
                thresh = threshold_otsu(resized_img)
                resized_img = resized_img <= thresh
            data.append(resized_img)
            
            if label not in fixed_labels_dict:
                fixed_labels_dict[label] = ind
                ind += 1
            labels.append(fixed_labels_dict[label])
        del tmp_data
    data = np.asarray(data)
    labels = np.asarray(labels)
    if verbose:
        print('Saving processed data in file')
    np.savez(DATA_PATH + 'train_full', data = data, labels = labels)
    np.save(DATA_PATH + 'fixed_labels', fixed_labels_dict)
    del data, labels
    gc.collect()
    return


# In[ ]:


fixed_labels_dict = {}
preprocess_data(DATA_FILES, fixed_labels_dict=fixed_labels_dict, verbose=1)


# Run the following cell to load full dataset (requires around 12Gb RAM):

# In[3]:


data = np.load(DATA_PATH + 'train_full.npz')
images = data['data']
labels = data['labels']


# In[4]:


fixed_labels_dict = np.load(DATA_PATH + 'fixed_labels.npy')


# In[5]:


fixed_labels_dict = fixed_labels_dict[()]


# In[6]:


def get_reversed_dict(dictionary):
    reversed_dict = {v: k for k, v in dictionary.items()}
    return reversed_dict


# In[7]:


class ConvNN():
    def __init__(self):
        self.model = Sequential()
        
    def init_model(self):
        self.model.add(Conv2D(64, (3, 3), input_shape=(68, 68, 1), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D((2,2), strides=(2,2)))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(AveragePooling2D((2,2), strides=(2,2)))
        self.model.add(BatchNormalization()) 
        
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1000, activation='softmax'))
        
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy']
                          )
    def print_model_info(self):
        self.model.summary()
    
    def get_model(self):
        return self.model
    
    def fit_model(self, x_train, x_valid, y_train, y_valid, batch_size=128, epochs=6):
        print('Fitting model on train data...')
        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(x_valid, y_valid)
                      )
        print('Saving model in file')
        self.model.save('./models_data/model.h5')
        
        print('Fitting model on validation data...')
        self.model.fit(x_valid,
                       y_valid,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1)
        
        print('Saving model in file')
        self.model.save('./models_data/model.h5')
        print('Successfully completed fitting process!')
        
    def predict(self, test_data):
        self.predicton_labels = np.argmax(self.model.predict(test_data, verbose=1), axis=1)
    
    def get_predictions(self):
        return self.predicton_labels
        
    def save_predictions_in_csv(self, reversed_labels_dict):
        ind = 1
        result = []
        for label in self.predicton_labels:
            result.append([ind, reversed_labels_dict[label]])
            ind += 1
        res_dataframe = pd.DataFrame(np.asarray(result))
        res_dataframe.to_csv(DATA_PATH + 'predictions.csv', index=False, header=('Id', 'Category'))


# In[8]:


def get_categorical_representation(data, num_classes=1000):
    return keras.utils.to_categorical(data, num_classes=num_classes)


# In[9]:


images_final = images.reshape(images.shape[0], 68, 68, 1)
labels_final = get_categorical_representation(labels)

get_ipython().run_line_magic('xdel', 'images')
get_ipython().run_line_magic('xdel', 'labels')
gc.collect()


# Requires extra 12Gb of RAM (~25Gb total by now)

# In[10]:


x_train, x_valid, y_train, y_valid = train_test_split(images_final, labels_final, test_size=0.15, random_state=17)


# In[11]:


cnn = ConvNN()
cnn.init_model()


# In[12]:


cnn.print_model_info()


# In[13]:


cnn.fit_model(x_train, x_valid, y_train, y_valid, batch_size=256, epochs=7)


# In[14]:


my_model = cnn.get_model()


# Deleting numpy objects from memory as they're no longer needed.

# In[15]:


get_ipython().run_line_magic('xdel', 'x_train')
get_ipython().run_line_magic('xdel', 'x_valid')
get_ipython().run_line_magic('xdel', 'y_train')
get_ipython().run_line_magic('xdel', 'y_valid')
gc.collect()


# -----------

# In[16]:


my_model.fit(images_final, labels_final, batch_size=512, epochs=1, verbose=1)


# In[17]:


my_model.save('./models_data/model.h5')


# In[18]:


my_model.fit(images_final, labels_final, batch_size=1024, epochs=1, verbose=1)


# In[19]:


my_model.save('./models_data/model.h5')


# -----------

# Let's prepare test data:

# In[20]:


test_data_np = np.load(DATA_PATH + 'test.npy')


# In[21]:


test_dataset = []
for img in tqdm(test_data_np):
    resized_img = resize(img, (68, 68), mode='constant')
    #thresh = threshold_otsu(resized_img)
    #binary_img = resized_img <= thresh
    test_dataset.append(resized_img)
test_dataset = np.asarray(test_dataset)
test_dataset = test_dataset.reshape(test_dataset.shape[0], 68, 68, 1)


# And make predictions:

# In[22]:


cnn.predict(test_dataset)


# In[23]:


reversed_labels_dict = get_reversed_dict(fixed_labels_dict)


# In[24]:


cnn.save_predictions_in_csv(reversed_labels_dict)

