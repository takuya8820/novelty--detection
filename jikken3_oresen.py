# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 10:51:52 2019

@author: takuya
"""

import pickle
import sys
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
if len(sys.argv) > 1:
	# noiseZ
    noisez = int(sys.argv[1])
	# noiseSigma
    if len(sys.argv) > 2:
        noiseSigma = int(sys.argv[2])
 
       """
jikkenPath = 'jikken'
jikkenPath2 = 'jikken2'
jikkenPath3 = 'jikken3'
jikkenvisualPath = 'jikkenkekka'


y1 = []
y2 = []
y3 = []
y4 = []
for late in range(5):
    noisez=128
    noiseSigma=128
    mx1 = []
    for targetChar in range(10):
        data1 = []
        for trialNo in range(1,4):
            postFix = "{}_{}".format(targetChar, trialNo)
            path1 = os.path.join(jikkenPath3,"noise{}_{}".format(noisez,noiseSigma))
            path = os.path.join(path1,"log{}.pickle".format(postFix))
            with open(path, "rb") as fp:
                batch = pickle.load(fp)
                batch_x_fake = pickle.load(fp)
                encoderR_train_value = pickle.load(fp)
                decoderR_train_value = pickle.load(fp)
                #encoderR_fake_train_value = pickle.load(fp)
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
                
                #dataは3回実験した結果を格納
            data1.append(f1DXs[late][14])
        #mxは各カテゴリの最大値を格納
        mx1.append(max(data1))
        
    s1=sum(mx1)
    n1=len(mx1)
    #meanは任意のnoiseSigmaとnoiseZのときのF値
    mean1=s1/n1
    y1.append(mean1)


for late in range(5):
    noisez=51
    noiseSigma=128
    mx2 = []
    for targetChar in range(10):
        data2 = []
        for trialNo in range(1,4):
            postFix = "{}_{}".format(targetChar, trialNo)
            path1 = os.path.join(jikkenPath3,"noise{}_{}".format(noisez,noiseSigma))
            path = os.path.join(path1,"log{}.pickle".format(postFix))
            with open(path, "rb") as fp:
                batch = pickle.load(fp)
                batch_x_fake = pickle.load(fp)
                encoderR_train_value = pickle.load(fp)
                decoderR_train_value = pickle.load(fp)
                #encoderR_fake_train_value = pickle.load(fp)
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
                
                #dataは3回実験した結果を格納
            data2.append(f1DRXs[late][14])
        #mxは各カテゴリの最大値を格納
        mx2.append(max(data2))
        
    s2=sum(mx2)
    n2=len(mx2)
    #meanは任意のnoiseSigmaとnoiseZのときのF値
   
    mean2=s2/n2

    y2.append(mean2)
    

for late in range(5):
    noiseSigma=128
    mx3 = []
    for targetChar in range(10):
        data3 = []
        for trialNo in range(1,4):
            postFix = "{}_{}".format(targetChar, trialNo)
            path1 = os.path.join(jikkenPath,"noise{}".format(noiseSigma))
            path = os.path.join(path1,"log{}.pickle".format(postFix))
            with open(path, "rb") as fp:
                batch = pickle.load(fp)
                batch_x_fake = pickle.load(fp)
                encoderR_train_value = pickle.load(fp)
                decoderR_train_value = pickle.load(fp)
                #encoderR_fake_train_value = pickle.load(fp)
                #decoderR_fake_train_value = pickle.load(fp)
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
                
                #dataは3回実験した結果を格納
            data3.append(f1DXs[late][14])
        #mxは各カテゴリの最大値を格納
        mx3.append(max(data3))
        
    s3=sum(mx3)
    n3=len(mx3)
    #meanは任意のnoiseSigmaとnoiseZのときのF値
   
    mean3=s3/n3

    y3.append(mean3)
    
for late in range(5):
    noiseSigma=13
    mx4 = []
    for targetChar in range(10):
        data4 = []
        for trialNo in range(1,4):
            postFix = "{}_{}".format(targetChar, trialNo)
            path1 = os.path.join(jikkenPath,"noise{}".format(noiseSigma))
            path = os.path.join(path1,"log{}.pickle".format(postFix))
            with open(path, "rb") as fp:
                batch = pickle.load(fp)
                batch_x_fake = pickle.load(fp)
                encoderR_train_value = pickle.load(fp)
                decoderR_train_value = pickle.load(fp)
                #encoderR_fake_train_value = pickle.load(fp)
                #decoderR_fake_train_value = pickle.load(fp)
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
                
                #dataは3回実験した結果を格納
            data4.append(f1DRXs[late][14])
        #mxは各カテゴリの最大値を格納
        mx4.append(max(data4))
        
    s4=sum(mx4)
    n4=len(mx4)
    #meanは任意のnoiseSigmaとnoiseZのときのF値
   
    mean4=s4/n4

    y4.append(mean4)    
    
    

 
sns.set()
sns.set_style('white')
sns.set_palette('Set1',2)
sns.set_context("paper",1.9)
    
x = ([10,20,30,40,50])

font = {'family' : 'YuGothic'}
plt.rc('font', **font)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x, y1, label='DODANFS D(X)', marker="o")
ax.plot(x, y2, label='DODANFS D(R(X))', marker="o")
ax.plot(x, y3, label='ALOCC D(X)')
ax.plot(x, y4, label='ALOCC D(R(X))')


ax.legend()
ax.set_xlabel("Percentage of outliers(%)")
ax.set_ylabel("F1-Score")
ax.set_xlim(10, 50)
ax.set_ylim(0, 1)

plt.show()
'''
path = os.path.join(jikkenvisualPath,"jikken.png")
plt.savefig(path)
'''