#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import cv2
import sys
import os
#if(lhs>rhs) then its no pneumonia


# In[3]:


xtrain_normal="G:\\documents\\kaggel datasets\\chest_xray\\train\\NORMAL"
xtrainnormal_path=os.path.join(xtrain_normal,'*g')
xtrainnormal_files=glob.glob(xtrainnormal_path)
a=[]
b=[]
c=[]
d=[]


# In[4]:


for f1 in xtrainnormal_files:
    if "IM" in f1:
     m=cv2.imread(f1,0)
     g=cv2.resize(m,dsize=(28,28),interpolation=cv2.INTER_CUBIC)
     a.append(g)
     b.append(0)
    else:
     m=cv2.imread(f1,0)
     g=cv2.resize(m,dsize=(28,28),interpolation=cv2.INTER_CUBIC)
     a.append(g)
     b.append(1)


# In[5]:


import numpy as np
import pandas as pd
x_train=np.array(a)
y_train=np.array(b)
print(y_train)


# In[6]:


print(x_train.shape)
print(y_train.shape)


# In[7]:


xtest_normal="G:\\documents\\kaggel datasets\\chest_xray\\test\\NORMAL"
xtestnormal_path=os.path.join(xtest_normal,'*g')
xtestnormal_files=glob.glob(xtestnormal_path)


# In[8]:




# In[9]:


import keras
from keras.layers import Dense,Dropout,Conv2D,Flatten,MaxPooling2D


# In[10]:


x_train=x_train.reshape(5216,28,28,1)


# In[12]:


model=keras.Sequential()
model.add(Conv2D(32,(6,6),strides=(3,3),padding='same',activation='sigmoid',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))







model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))


# In[13]:


model.summary()


# model.compile(optimizer='adam'

# In[14]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[15]:


from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)
print(y_train)


# In[17]:


model.fit(x_train,y_train,epochs=50)


# In[18]:


x_test=np.array(c)
y_test=np.array(d)

print(len(x_test))
                


# In[19]:


x_test=x_test.reshape(624,28,28,1)
y_test=to_categorical(y_test)


# In[20]:


loss,acc=model.evaluate(x_test,y_test)


# In[21]:


loss,acc


# In[22]:


p=model.predict(x_test)


# In[23]:


print(p)


# In[24]:


print(p)


# In[25]:



valid='G:\\documents\\kaggel datasets\\chest_xray\\val\\NORMAL'
valid_path=os.path.join(valid,'*g')
valid_files=glob.glob(valid_path)


# In[26]:


aa=[]
for f3 in valid_files:
    
     mm=cv2.imread(f3,0)
     gg=cv2.resize(mm,dsize=(28,28),interpolation=cv2.INTER_CUBIC)
     aa.append(gg)
     


# In[27]:


valid_test=np.array(aa)
valid_test=valid_test.reshape(8,28,28,1)


# In[28]:


unseen=model.predict(valid_test)




# In[31]:


print("sample output")

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for hh in range(0,8):
    plt.subplot(2,4,hh+1)
    rg=np.argmax(unseen[hh])
    plt.xlabel('true is normal\n predicted is %s'%classes[rg])
    rf=cv2.imread(valid_files[hh])
    plt.imshow(rf)
    


# In[32]:


gh='G:\\documents\\kaggel datasets\\chest_xray\\val\\PNEUMONIA'
hg=os.path.join(gh,'*g')
files=glob.glob(hg)

print(files)

# In[33]:


csk=[]
for f4 in files:
   
     mi=cv2.imread(f4,0)
     kk=cv2.resize(mi,dsize=(28,28),interpolation=cv2.INTER_CUBIC)
     csk.append(kk)
     


# In[34]:


valid_set=np.array(csk)
valid_set=valid_set.reshape(8,28,28,1)


# In[35]:


fin=model.predict(valid_set)


# In[36]:


for j in range(0,7):
  print(np.argmax(fin[j]))


# In[37]:


print("sample output")

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for hh in range(0,8):
    plt.subplot(2,4,hh+1)
    gr=np.argmax(fin[hh])
    
    plt.xlabel('true is pneumonia\n predicted is %s'%classes[gr])
    fr=cv2.imread(files[hh])
    plt.imshow(fr)


# In[38]:


print(fin)


# In[39]:


print(unseen)


# In[40]:


ree=cv2.imread('shubhamxray.jpg',0)


# In[41]:


print(ree.shape)


# In[44]:


re=cv2.resize(ree,dsize=(28,28))


# In[45]:


print(re.shape)


# In[46]:


re=re.reshape(1,28,28,1)


# In[47]:


yr=model.predict(re)


# In[48]:


print(yr)


# In[ ]:




