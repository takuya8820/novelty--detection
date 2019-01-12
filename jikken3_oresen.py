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

if len(sys.argv) > 1:
	# noiseZ
    noisez = int(sys.argv[1])
	# noiseSigma
    if len(sys.argv) > 2:
        noiseSigma = int(sys.argv[2])

jikkenPath3 = 'jikken3'

data = []
mx1 = []
mx2 = []
y1 = []
y2 = []
for late in range(5):
    mx1 = []
    mx2 = []
    for targetChar in range(10):
        data1 = []
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
            data1.append(precisionDXs[late][14])
            data2.append(precisionDRXs[late][14])
        #mxは各カテゴリの最大値を格納
        mx1.append(max(data1))
        mx2.append(max(data2))
        
    s1=sum(mx1)
    n1=len(mx1)
    s2=sum(mx2)
    n2=len(mx2)
    #meanは任意のnoiseSigmaとnoiseZのときのF値
    mean1=s1/n1
    mean2=s2/n2
    y1.append(mean1)
    y2.append(mean2)
    
sns.set()
sns.set_style('white')
sns.set_palette('Set1')
sns.set_context("paper")
    
x = ([10,20,30,40,50])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(x, y1, label='D(X)')
ax.plot(x, y2, label='D(R(X))')

ax.legend()
ax.set_xlabel("Percentage of outliers(%)")
ax.set_ylabel("F1-Score")
ax.set_xlim(10, 50)
ax.set_ylim(0, 1)

plt.show()