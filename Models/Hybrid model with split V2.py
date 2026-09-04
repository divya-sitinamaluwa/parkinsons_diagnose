#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


train_data1 = []
train_label1 = [] 

train_data2 = []
train_label2 = [] 


# In[3]:


for directory_path in glob.glob("Dataset/Split Image/Data vertical 2/Training Data/Group 1/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data1.append(img)
        train_label1.append(label)


# In[4]:


train_data1 = np.array(train_data1)
train_label1 = np.array(train_label1)


# In[5]:


for directory_path in glob.glob("Dataset/Split Image/Data vertical 2/Training Data/Group 2/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data2.append(img)
        train_label2.append(label)


# In[6]:


train_data2 = np.array(train_data2)
train_label2 = np.array(train_label2)


# In[7]:


val_data1 = []
val_label1 = [] 

val_data2 = []
val_label2 = [] 


# In[8]:


for directory_path in glob.glob("Dataset/Split Image/Data vertical 2/Validation Data/Group 1/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data1.append(img)
        val_label1.append(label)


# In[9]:


val_data1 = np.array(val_data1)
val_label1 = np.array(val_label1)


# In[10]:


# CORRECTED: this used to point at "Group 1" again (copy-paste bug), so val_data2 was
# actually a duplicate of val_data1, not genuine Group-2 validation data. Fixed to Group 2.
for directory_path in glob.glob("Dataset/Split Image/Data vertical 2/Validation Data/Group 2/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data2.append(img)
        val_label2.append(label)


# In[11]:


val_data2 = np.array(val_data2)
val_labels2 = np.array(val_label2)


# In[11b]: CORRECTED - load the held-out test set for both groups (folders existed but were
# never read before)

test_data1 = []
test_label1 = []
test_data2 = []
test_label2 = []

for directory_path in glob.glob("Dataset/Split Image/Data vertical 2/Testing Data/Group 1/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data1.append(img)
        test_label1.append(label)

for directory_path in glob.glob("Dataset/Split Image/Data vertical 2/Testing Data/Group 2/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data2.append(img)
        test_label2.append(label)

test_data1 = np.array(test_data1)
test_label1 = np.array(test_label1)
test_data2 = np.array(test_data2)
test_label2 = np.array(test_label2)


# In[12]:


train_label = np.concatenate((train_label1, train_label2))


# In[13]:


val_label = np.concatenate((val_label1, val_label2))


# In[13b]: CORRECTED - concatenate test labels the same way

test_label = np.concatenate((test_label1, test_label2))


# In[14]:


le = preprocessing.LabelEncoder()


# In[15]:


le.fit(train_label)
train_label = le.transform(train_label)


# In[16]:


le.fit(val_label)
val_label = le.transform(val_label)


# In[16b]: CORRECTED - encode test labels the same way

le.fit(test_label)
test_label = le.transform(test_label)


# In[17]:


print(train_data1.shape)

print(train_data2.shape)


# In[18]:


print(val_data1.shape)

print(val_data2.shape)


# In[19]:


train_data1 = train_data1.reshape(train_data1.shape[0],train_data1.shape[2],train_data1.shape[1],train_data1.shape[3])

train_data2 = train_data2.reshape(train_data2.shape[0],train_data2.shape[2],train_data2.shape[1],train_data2.shape[3])


# In[20]:


val_data1 = val_data1.reshape(val_data1.shape[0],val_data1.shape[2],val_data1.shape[1],val_data1.shape[3])

val_data2 = val_data2.reshape(val_data2.shape[0],val_data2.shape[2],val_data2.shape[1],val_data2.shape[3])


# In[20b]: CORRECTED - reshape the test data groups the same way as train/val

test_data1 = test_data1.reshape(test_data1.shape[0],test_data1.shape[2],test_data1.shape[1],test_data1.shape[3])

test_data2 = test_data2.reshape(test_data2.shape[0],test_data2.shape[2],test_data2.shape[1],test_data2.shape[3])


# In[21]:


print(train_data1.shape)

print(train_data2.shape)


# In[22]:


print(val_data1.shape)

print(val_data2.shape)


# In[23]:


vgg_model1 = VGG16(input_shape=(143, 188,3), weights='imagenet', include_top=False, pooling = 'avg')

vgg_model2 = VGG16(input_shape=(143, 188,3), weights='imagenet', include_top=False, pooling = 'avg')


# In[24]:


for layer in vgg_model1.layers:
    layer.trainable = False


# In[25]:


for layer in vgg_model2.layers:
    layer.trainable = False


# In[26]:


vgg_model1.summary()


# In[27]:


train_features1 = vgg_model1.predict(train_data1, verbose= 1)


# In[28]:


val_features1 = vgg_model1.predict(val_data1, verbose= 1)


# In[29]:


print(train_features1.shape)
print(val_features1.shape)


# In[30]:


train_features2 = vgg_model2.predict(train_data2, verbose= 1)


# In[31]:


val_features2 = vgg_model2.predict(val_data2, verbose= 1)


# In[31b]: CORRECTED - extract test-set features for both groups

test_features1 = vgg_model1.predict(test_data1, verbose= 1)
test_features2 = vgg_model2.predict(test_data2, verbose= 1)


# In[32]:


print(train_features2.shape)
print(val_features2.shape)


# In[33]:


train_features = np.concatenate((train_features1, train_features2))


# In[34]:


val_features = np.concatenate((val_features1, val_features2))


# In[34b]: CORRECTED - concatenate test features the same way

test_features = np.concatenate((test_features1, test_features2))


# In[35]:


print(train_features.shape)
print(val_features.shape)
print(test_features.shape)


# In[ ]:





# In[ ]:





# ------------------------------------

# In[36]:


classifier = Sequential()

classifier.add(layers.Dense(512, activation='ReLU', input_dim = 512))
classifier.add(layers.Dropout(0.8))
classifier.add(layers.Dense(1, activation='sigmoid'))


# In[37]:


classifier.compile(
  loss='binary_crossentropy',
  optimizer='Adam',
  metrics=['accuracy']
)


# In[38]:


history = classifier.fit(train_features, train_label, epochs=50,  validation_data= (val_features, val_label))


# In[39]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = classifier.evaluate(val_features, verbose=2)


# In[40]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[41]:


predict = classifier.predict(val_features)


# In[42]:


val_predict = ((predict > 0.5)+0).ravel()


# In[43]:


print(classification_report(val_label, val_predict))


# In[44]:


print(confusion_matrix(val_label, val_predict))


# In[45]:


train_loss, train_acc = classifier.evaluate(train_features, verbose=2)
print(train_acc)

val_loss, val_acc = classifier.evaluate(val_features, verbose=2)
print(val_acc)


# In[ ]: CORRECTED - evaluate ONCE on the held-out test set; report this number, not val_acc.

test_loss, test_acc = classifier.evaluate(test_features, verbose=2)
print("Held-out TEST accuracy (report this, not val_acc):", test_acc)

test_predict_probs = classifier.predict(test_features)
test_predict = ((test_predict_probs > 0.5) + 0).ravel()

print("Held-out TEST classification report:")
print(classification_report(test_label, test_predict))

print("Held-out TEST confusion matrix:")
print(confusion_matrix(test_label, test_predict))


# In[ ]:




