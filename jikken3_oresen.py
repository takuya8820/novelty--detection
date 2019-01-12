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

if len(sys.argv) > 1:
	# noiseZ
    noisez = int(sys.argv[1])
	# noiseSigma
    if len(sys.argv) > 2:
        noiseSigma = int(sys.argv[2])

jikkenPath3 = 'jikken3'

data = []
mx = []
x = []
y = []
for late in range(4):
    for targetChar in range(10):
        data = []
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
            data.append(precisionDXs[late][14])
                #mxは各カテゴリの最大値を格納
        mx.append(max(data))
    
    s=sum(mx)
    n=len(mx)
    #meanは任意のnoiseSigmaとnoiseZのときのF値
    mean=s/n
    x.append(late)
    y.append(mean)
    pdb.set_trace()
    