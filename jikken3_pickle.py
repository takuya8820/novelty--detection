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
	# 文字の種類
    targetChar = int(sys.argv[1])
	# trail no.
    if len(sys.argv) > 2:
        trialNo = int(sys.argv[2])
        # noiseSigma
        if len(sys.argv) > 3:
            noisez = int(sys.argv[3])
            #late
            if len(sys.argv) > 4:
                    late = int(sys.argv[4])
            
noiseSigma = 51
threSquaredLoss = 200
logPath = 'logs'
jikkenPath = 'jikken'
jikkenPath3 = 'jikken3'

postFix = "{}_{}".format(targetChar, trialNo)

#path1 = os.path.join(logPath,"noiseSigma_{}".format(noiseSigma))
path1 = os.path.join(jikkenPath3,"noiseZ_{}".format(noisez))
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
    
print(precisionDXs[late][14])
print(precisionDRXs[late][14])
'''    
print(encoderR_train_value)
#zと何を比較して散布図に乗せればよい？
plt.scatter(encoderR_train_value)
plt.show()
'''