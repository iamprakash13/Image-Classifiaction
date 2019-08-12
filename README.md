Classiﬁcation   of Images of Childhood Pneumonia using   Convolutional Neural Networks

Keywords: Pneumonia, X-Ray, CNN.
Abstract: In this paper we describe a comparative classiﬁcation of Pneumonia using Convolution Neural Network. The database used was Chest X-Ray Images for Classiﬁcation made available by kaggel with a total of 5216 train and 624 test images , with 2 classes: normal and pneumonia.

1) INTRODUCTION:
In this article, I will be using CNN to train on a number of chest X-rays and predict whether pneumonia is present or not. This problem is different than what we saw in part 1. In this article, we will be dealing with gray-scale chest X-ray images while in part 1, we had to train CNN on colored blood cell images. The problem also becomes a bit challenging as pneumonia usually is represented as opacity in the lobes of lungs.
2) Data Source:
The dataset is organized into 3 folders (train, test) and contains sub-folders for each image category (Pneumonia/Normal). There are 5840 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.
For the analysis of chest x-ray images, all chest radio-graphs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

3)CODE DESCRIPTION:

Importing Necessary Libraries:
import glob
import cv2
import sys
import os
Images in Train and Test folders
1.	Train folder: (5216 images in train normal folder)
2.	Test folder: (624 images in test normal folder )
Path of our dataset:
xtrain_normal="G:\\documents\\kaggel datasets\\chest_xray\\train\\NORMAL"
xtrainnormal_path=os.path.join(xtrain_normal,'*g')
xtrainnormal_files=glob.glob(xtrainnormal_path)
a=[]
b=[]
c=[]
d=[]
RESIZING ALL images to 28*28 PIXELS AND CONVERTED INTO GRAYSCALE AND LABEL THE IMAGES OF 2 CLASSES:
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
import numpy as np
import pandas as pd
x_train=np.array(a)
y_train=np.array(b)
print(y_train)
print(x_train.shape)

print(y_train.shape)
xtest_normal="G:\\documents\\kaggel datasets\\chest_xray\\test\\NORMAL"
xtestnormal_path=os.path.join(xtest_normal,'*g')
xtestnormal_files=glob.glob(xtestnormal_path)
for f2 in xtestnormal_files:
    if "IM" in f2:
     n=cv2.imread(f2,0)
     h=cv2.resize(n,dsize=(28,28))
     c.append(h)
     d.append(0)
    else:
     n=cv2.imread(f2,0)
     h=cv2.resize(n,dsize=(28,28))
     c.append(h)
     d.append(1)
BUILDING CNN MODEL FOR CLASSIFICATION USING KERAS LIBRARY:
import keras
from keras.layers import Dense,Dropout,Conv2D,Flatten,MaxPooling2D
x_train=x_train.reshape(5216,28,28,1)
model=keras.Sequential()
model.add(Conv2D(32,(6,6),strides=(3,3),padding='same',activation='sigmoid',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))
ENCODE THE IMAGE DATA:
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)
print(y_train)
COMPILING MODEL:
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
TRAIN RHE MODEL FOR 50 EPOCHS:
model.fit(x_train,y_train,epochs=50)
After training I got an accuracy of around 93% on training data
ACCURACY OF TESTING DATA:
x_test=np.array(c)
y_test=np.array(d)

print(len(x_test))
x_test=x_test.reshape(624,28,28,1)
y_test=to_categorical(y_test)
loss,acc=model.evaluate(x_test,y_test)
loss,acc
I got accuracy of 74% on testing Data


SAMPLE OUTPUT:

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for hh in range(0,8):
    plt.subplot(2,4,hh+1)
    gr=np.argmax(fin[hh])
    
    plt.xlabel('true is pneumonia\n predicted is %s'%classes[gr])
    fr=cv2.imread(files[hh])
    plt.imshow(fr)
                 

Conclusion:
In this paper we have demonstrate work of detection and classiﬁcation of images for the detection of pneumonia from the chest X-ray of patients. The Convolutional Neural Network was used to train the neural network and, for the validation of the model, Cross validation was used. The classiﬁcation model presented was efﬁcient in the classiﬁcation, obtaining an average accuracy of 74 % in the tests against 93% on training data.
REFERENCES:
https://medium.com/@abhikjha/medical-images-use-cases-of-deep-learning-part-2-c5df16f77ab1



