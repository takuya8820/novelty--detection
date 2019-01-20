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
	# noiseSigma
    noiseSigma = int(sys.argv[1])
	# late
    if len(sys.argv) > 2:
        late = int(sys.argv[2])
        
            
            
threSquaredLoss = 200
logPath = 'logs'
jikkenPath = 'jikken'
jikkenPath2 = 'jikken2'

#maxf = [[] for tmp in np.arange(10)]
#precision = [[] for tmp in np.arange(4)]
data = []
mx = []
for targetChar in range(10):
    print(targetChar)
    data = []
    for trialNo in range(1,4):
        postFix = "{}_{}".format(targetChar, trialNo)
        path1 = os.path.join(jikkenPath,"noise{}".format(noiseSigma))
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
            pdb.set_trace()
            print(precisionDRXs[late][14])
"""
        data.append(precisionDRXs[late][14])
    mx.append(max(data))
s=sum(mx)
n=len(mx)
mean=s/n
print(mean)
print(lossR_values)
"""

