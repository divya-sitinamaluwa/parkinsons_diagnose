#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout,MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from keras.applications.imagenet_utils import decode_predictions 
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report


# In[3]:


train_data = []
train_labels = [] 


# In[4]:


for directory_path in glob.glob("Dataset/Full Image/Training Data/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (289, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data.append(img)
        train_labels.append(label)


# In[5]:


train_data = np.array(train_data)
train_labels = np.array(train_labels)


# In[6]:


val_data = []
val_labels = [] 


# In[7]:


for directory_path in glob.glob("Dataset/Full Image/Validation Data/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (289, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data.append(img)
        val_labels.append(label)


# In[8]:


val_data = np.array(val_data)
val_labels = np.array(val_labels)


# In[8b]: CORRECTED - load the held-out test set. This folder existed on disk but was never
# read by this script before, so every reported metric was actually a validation metric.

test_data = []
test_labels = []

for directory_path in glob.glob("Dataset/Full Image/Testing Data/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (289, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data.append(img)
        test_labels.append(label)

test_data = np.array(test_data)
test_labels = np.array(test_labels)


# In[9]:


le = preprocessing.LabelEncoder()


# In[10]:


le.fit(train_labels)
train_label = le.transform(train_labels)


# In[11]:


le.fit(val_labels)
val_label = le.transform(val_labels)


# In[11b]: CORRECTED - encode test labels the same way

le.fit(test_labels)
test_label = le.transform(test_labels)


# In[12]:


train_data.shape


# In[13]:


train_data = train_data.reshape(train_data.shape[0],train_data.shape[2],train_data.shape[1],train_data.shape[3])


# In[14]:


val_data = val_data.reshape(val_data.shape[0],val_data.shape[2],val_data.shape[1],val_data.shape[3])


# In[14b]: CORRECTED - reshape the test set the same way as train/val

test_data = test_data.reshape(test_data.shape[0],test_data.shape[2],test_data.shape[1],test_data.shape[3])


# In[15]:


train_data.shape


# In[16]:


cnn_model = Sequential()

cnn_model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(289, 188, 3)))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(layers.MaxPooling2D((2, 2)))

cnn_model.add(layers.Flatten())
cnn_model.add(layers.Dense(512, activation='relu'))
cnn_model.add(layers.Dense(1, activation='sigmoid'))


# In[17]:


cnn_model.compile(
  loss='binary_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[18]:


history = cnn_model.fit(train_data, train_label, epochs=20,  validation_data= (val_data, val_label))


# In[19]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = cnn_model.evaluate(val_data, verbose=2)


# In[20]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[21]:


predict = cnn_model.predict(val_data)


# In[22]:


val_predict = ((predict > 0.5)+0).ravel()


# In[23]:


print(classification_report(val_label, val_predict))


# In[24]:


print(confusion_matrix(val_label, val_predict))


# In[26]:


train_loss, train_acc = cnn_model.evaluate(train_data, verbose=2)
print(train_acc)

val_loss, val_acc = cnn_model.evaluate(val_data, verbose=2)
print(val_acc)


# In[27]: CORRECTED - evaluate ONCE on the held-out test set. Use the validation numbers above
# only for model/hyperparameter selection during development; THIS is the number that should
# be reported in the paper as the model's performance.

test_loss, test_acc = cnn_model.evaluate(test_data, verbose=2)
print("Held-out TEST accuracy (report this, not val_acc):", test_acc)

test_predict_probs = cnn_model.predict(test_data)
test_predict = ((test_predict_probs > 0.5) + 0).ravel()

print("Held-out TEST classification report:")
print(classification_report(test_label, test_predict))

print("Held-out TEST confusion matrix:")
print(confusion_matrix(test_label, test_predict))


# In[ ]:




