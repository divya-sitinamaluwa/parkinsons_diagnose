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

train_data3 = []
train_label3 = [] 

train_data4 = []
train_label4 = [] 


# In[3]:


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Training Data/Group 1/*"):
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


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Training Data/Group 2/*"):
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


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Training Data/Group 3/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data3.append(img)
        train_label3.append(label)


# In[8]:


train_data3 = np.array(train_data3)
train_label3 = np.array(train_label3)


# In[9]:


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Training Data/Group 4/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_data4.append(img)
        train_label4.append(label)


# In[10]:


train_data4 = np.array(train_data4)
train_label4 = np.array(train_label4)


# In[31]:


val_data1 = []
val_label1 = [] 

val_data2 = []
val_label2 = [] 

val_data3 = []
val_label3 = [] 

val_data4 = []
val_label4 = [] 


# In[32]:


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Validation Data/Group 1/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data1.append(img)
        val_label1.append(label)


# In[33]:


val_data1 = np.array(val_data1)
val_label1 = np.array(val_label1)


# In[34]:


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Validation Data/Group 2/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data2.append(img)
        val_label2.append(label)


# In[35]:


val_data2 = np.array(val_data2)
val_labels2 = np.array(val_label2)


# In[36]:


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Validation Data/Group 3/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data3.append(img)
        val_label3.append(label)


# In[37]:


val_data3 = np.array(val_data3)
val_labels3 = np.array(val_label3)


# In[38]:


for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Validation Data/Group 4/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        val_data4.append(img)
        val_label4.append(label)


# In[39]:


val_data4 = np.array(val_data4)
val_labels4 = np.array(val_label4)


# In[39b]: CORRECTED - load the held-out test set for all 4 groups (folders existed but were
# never read before).

test_data1 = []
test_label1 = []
test_data2 = []
test_label2 = []
test_data3 = []
test_label3 = []
test_data4 = []
test_label4 = []

for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Testing Data/Group 1/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data1.append(img)
        test_label1.append(label)

for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Testing Data/Group 2/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data2.append(img)
        test_label2.append(label)

for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Testing Data/Group 3/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data3.append(img)
        test_label3.append(label)

for directory_path in glob.glob("Dataset/Split Image/Data horizontal 4/Testing Data/Group 4/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (143, 188))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_data4.append(img)
        test_label4.append(label)

test_data1 = np.array(test_data1)
test_label1 = np.array(test_label1)
test_data2 = np.array(test_data2)
test_label2 = np.array(test_label2)
test_data3 = np.array(test_data3)
test_label3 = np.array(test_label3)
test_data4 = np.array(test_data4)
test_label4 = np.array(test_label4)


# In[40]:


train_label = np.concatenate((train_label1, train_label2, train_label3, train_label4))


# In[41]:


val_label = np.concatenate((val_label1, val_label2, val_label3, val_label4))


# In[41b]: CORRECTED - concatenate the test labels the same way

test_label = np.concatenate((test_label1, test_label2, test_label3, test_label4))


# In[42]:


le = preprocessing.LabelEncoder()


# In[43]:


le.fit(train_label)
train_label = le.transform(train_label)


# In[44]:


le.fit(val_label)
val_label = le.transform(val_label)


# In[44b]: CORRECTED - encode test labels the same way

le.fit(test_label)
test_label = le.transform(test_label)


# In[60]:


print(train_data1.shape)

print(train_data2.shape)

print(train_data3.shape)

print(train_data3.shape)


# In[61]:


print(val_data1.shape)

print(val_data2.shape)

print(val_data3.shape)

print(val_data4.shape)


# In[62]:


train_data1 = train_data1.reshape(train_data1.shape[0],train_data1.shape[2],train_data1.shape[1],train_data1.shape[3])

train_data2 = train_data2.reshape(train_data2.shape[0],train_data2.shape[2],train_data2.shape[1],train_data2.shape[3])

train_data3 = train_data3.reshape(train_data3.shape[0],train_data3.shape[2],train_data3.shape[1],train_data3.shape[3])

train_data4 = train_data4.reshape(train_data4.shape[0],train_data4.shape[2],train_data4.shape[1],train_data4.shape[3])


# In[66]:


val_data1 = val_data1.reshape(val_data1.shape[0],val_data1.shape[2],val_data1.shape[1],val_data1.shape[3])

val_data2 = val_data2.reshape(val_data2.shape[0],val_data2.shape[2],val_data2.shape[1],val_data2.shape[3])

val_data3 = val_data3.reshape(val_data3.shape[0],val_data3.shape[2],val_data3.shape[1],val_data3.shape[3])

val_data4 = val_data4.reshape(val_data4.shape[0],val_data4.shape[2],val_data4.shape[1],val_data4.shape[3])


# In[66b]: CORRECTED - reshape the test data groups the same way as train/val

test_data1 = test_data1.reshape(test_data1.shape[0],test_data1.shape[2],test_data1.shape[1],test_data1.shape[3])

test_data2 = test_data2.reshape(test_data2.shape[0],test_data2.shape[2],test_data2.shape[1],test_data2.shape[3])

test_data3 = test_data3.reshape(test_data3.shape[0],test_data3.shape[2],test_data3.shape[1],test_data3.shape[3])

test_data4 = test_data4.reshape(test_data4.shape[0],test_data4.shape[2],test_data4.shape[1],test_data4.shape[3])


# In[67]:


print(train_data1.shape)

print(train_data2.shape)

print(train_data3.shape)

print(train_data3.shape)


# In[68]:


print(val_data1.shape)

print(val_data2.shape)

print(val_data3.shape)

print(val_data4.shape)


# In[69]:


vgg_model1 = VGG16(input_shape=(143, 188,3), weights='imagenet', include_top=False, pooling = 'avg')

vgg_model2 = VGG16(input_shape=(143, 188,3), weights='imagenet', include_top=False, pooling = 'avg')

vgg_model3 = VGG16(input_shape=(143, 188,3), weights='imagenet', include_top=False, pooling = 'avg')

vgg_model4 = VGG16(input_shape=(143, 188,3), weights='imagenet', include_top=False, pooling = 'avg')


# In[70]:


for layer in vgg_model1.layers:
    layer.trainable = False


# In[71]:


for layer in vgg_model2.layers:
    layer.trainable = False


# In[72]:


for layer in vgg_model3.layers:
    layer.trainable = False


# In[73]:


for layer in vgg_model4.layers:
    layer.trainable = False


# In[74]:


vgg_model1.summary()


# In[75]:


train_features1 = vgg_model1.predict(train_data1, verbose= 1)


# In[76]:


val_features1 = vgg_model1.predict(val_data1, verbose= 1)


# In[77]:


print(train_features1.shape)
print(val_features1.shape)


# In[78]:


train_features2 = vgg_model2.predict(train_data2, verbose= 1)


# In[79]:


val_features2 = vgg_model2.predict(val_data2, verbose= 1)


# In[80]:


print(train_features2.shape)
print(val_features2.shape)


# In[81]:


train_features3 = vgg_model3.predict(train_data3, verbose= 1)


# In[82]:


val_features3 = vgg_model3.predict(val_data3, verbose= 1)


# In[83]:


print(train_features3.shape)
print(val_features3.shape)


# In[84]:


train_features4 = vgg_model4.predict(train_data4, verbose= 1)


# In[85]:


val_features4 = vgg_model4.predict(val_data4, verbose= 1)


# In[85b]: CORRECTED - extract test-set features for all 4 groups

test_features1 = vgg_model1.predict(test_data1, verbose= 1)
test_features2 = vgg_model2.predict(test_data2, verbose= 1)
test_features3 = vgg_model3.predict(test_data3, verbose= 1)
test_features4 = vgg_model4.predict(test_data4, verbose= 1)


# In[86]:


print(train_features4.shape)
print(val_features4.shape)


# In[87]:


train_features = np.concatenate((train_features1, train_features2, train_features3, train_features4))


# In[88]:


val_features = np.concatenate((val_features1, val_features2, val_features3, val_features4))


# In[88b]: CORRECTED - concatenate test features the same way

test_features = np.concatenate((test_features1, test_features2, test_features3, test_features4))


# In[89]:


print(train_features.shape)
print(val_features.shape)
print(test_features.shape)


# In[ ]:





# In[ ]:





# ------------------------------------

# In[90]:


classifier = Sequential()

classifier.add(layers.Dense(512, activation='ReLU', input_dim = 512))
classifier.add(layers.Dropout(0.8))
classifier.add(layers.Dense(1, activation='sigmoid'))


# In[91]:


classifier.compile(
  loss='binary_crossentropy',
  optimizer='Adam',
  metrics=['accuracy']
)


# In[92]:


history = classifier.fit(train_features, train_label, epochs=50,  validation_data= (val_features, val_label))


# In[93]:


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = classifier.evaluate(val_features, verbose=2)


# In[94]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# In[95]:


predict = classifier.predict(val_features)


# In[96]:


val_predict = ((predict > 0.5)+0).ravel()


# In[97]:


print(classification_report(val_label, val_predict))


# In[98]:


print(confusion_matrix(val_label, val_predict))


# In[99]:


train_loss, train_acc = classifier.evaluate(train_features, verbose=2)
print(train_acc)

val_loss, val_acc = classifier.evaluate(val_features, verbose=2)
print(val_acc)


# In[ ]: CORRECTED - evaluate ONCE on the held-out test set; report this number, not val_acc.
# Table 3 currently names this configuration (split into 4, average pooling, ReLU, Adam,
# 512 neurons, 0.5 dropout) as the "best" - this is the trustworthy number for that claim.
# Note: this script currently uses 0.8 dropout in the classifier above, not the 0.5 dropout
# recorded in Table 3 - worth double-checking against the manuscript before re-reporting.

test_loss, test_acc = classifier.evaluate(test_features, verbose=2)
print("Held-out TEST accuracy (report this, not val_acc):", test_acc)

test_predict_probs = classifier.predict(test_features)
test_predict = ((test_predict_probs > 0.5) + 0).ravel()

print("Held-out TEST classification report:")
print(classification_report(test_label, test_predict))

print("Held-out TEST confusion matrix:")
print(confusion_matrix(test_label, test_predict))


# In[ ]:




