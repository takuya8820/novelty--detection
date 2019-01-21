# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:11:32 2019

@author: takuya
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 09:55:34 2018

@author: takuya
"""

import pickle
import sys
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) > 1:
	# noiseZ
    targetChar = int(sys.argv[1])
	# noiseSigma
    if len(sys.argv) > 2:
        trialNo = int(sys.argv[2])
  

jikkenPath3 = 'jikken3'
noisez=51
noiseSigma=128


postFix = "{}_{}".format(targetChar, trialNo)

path1 = os.path.join(jikkenPath3,"noise{}_{}".format(noisez,noiseSigma))
path = os.path.join(path1,"log{}.pickle".format(postFix))
with open(path, "rb") as fp:
    batch = pickle.load(fp)
    batch_x_fake = pickle.load(fp)
    encoderR_train_value = pickle.load(fp)
    decoderR_train_value = pickle.load(fp)
    encoderR_fake_train_value = pickle.load(fp)
    decoderR_fake_train_value = pickle.load(fp)
    predictFake_train_value = pickle.load(fp)
    predictTrue_train_value = pickle.load(fp)
    test_x = pickle.load(fp)
    test_y = pickle.load(fp)
    decoderR_test_value = pickle.load(fp)
    predictDX_value = pickle.load(fp)
    predictDRX_value = pickle.load(fp)
    recallDXs = pickle.load(fp)
    precisionDXs = pickle.load(fp)
    f1DXs = pickle.load(fp)
    recallDRXs = pickle.load(fp)
    precisionDRXs = pickle.load(fp)
    f1DRXs = pickle.load(fp)
    lossR_values = pickle.load(fp)
    lossRAll_values = pickle.load(fp)
    lossD_values = pickle.load(fp)
    params = pickle.load(fp)

#------データのプロット----------
    
#Zでノイズを付加したデータ
x1=encoderR_fake_train_value[:,0]
y1=encoderR_fake_train_value[:,1]

#Zでノイズを付加していないデータ
x2=encoderR_train_value[:,0]
y2=encoderR_train_value[:,1]

'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(x1, y1, c='red')
ax.scatter(x2, y2, c='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
'''

plt.scatter(x1, y1, c="red")
plt.scatter(x2, y2, c="blue")
#---------------------------

#-------画像の表示------------

for i in range(5):
    a1=decoderR_fake_train_value[i]
    b1=decoderR_train_value[i]
    
    plt.imshow(a1)
    plt.imshow(b1)


plt.show()




