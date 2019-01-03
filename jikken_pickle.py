# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:12:09 2018

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
	# noiseSigma
    if len(sys.argv) > 2:
        noiseSigma = int(sys.argv[2])
        # late
        if len(sys.argv) > 3:
            late = int(sys.argv[3])
            
            
            
threSquaredLoss = 200
logPath = 'logs'
jikkenPath = 'jikken'
jikkenPath2 = 'jikken2'

#postFix = "{}_{}".format(targetChar, trialNo)
precision = [[] for tmp in np.arange(4)]

for trialNo in range(1,4):
    postFix = "{}_{}".format(targetChar, trialNo)
    path1 = os.path.join(jikkenPath,"noiseSigma{}".format(noiseSigma))
    path = os.path.join(path1,"log{}.pickle".format(postFix))
    with open(path, "rb") as fp:
        batch = pickle.load(fp)
        batch_x_fake = pickle.load(fp)
        encoderR_train_value = pickle.load(fp)
        decoderR_train_value = pickle.load(fp)
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
        
        precision[trialNo].append(precisionDXs[late][14])

mx = max(precision)
print(mx)


#print(precisionDRXs[late][14])

'''    
print(encoderR_train_value)
#zと何を比較して散布図に乗せればよい？
plt.scatter(encoderR_train_value)
plt.show()
'''