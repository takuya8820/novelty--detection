# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:35:29 2019

@author: takuya
"""
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import math, os
import pickle
import pdb
#import input_data
import matplotlib.pylab as plt
import sys
from sklearn.decomposition import PCA


if len(sys.argv) > 1:
	# 文字の種類
    targetChar = int(sys.argv[1])
	# trail no.
    if len(sys.argv) > 2:
        trialNo = int(sys.argv[2])   
        if len(sys.argv) > 3:
            noisez = int(sys.argv[3])
            if len(sys.argv) > 4:
                 noiseSigma = int(sys.argv[4])
            

sess = tf.Session()
sess.run(tf.global_variables_initializer())

jikkenPath3 = 'jikken3'

postFix = "{}_{}".format(targetChar, trialNo)
path1 = os.path.join(jikkenPath3,"noise{}_{}".format(noisez,noiseSigma))
path = os.path.join(path1,"log{}.pickle".format(postFix))
with open(path, "rb") as fp:
    batch = pickle.load(fp)
    batch_x_fake = pickle.load(fp)
    encoderR_train_value = pickle.load(fp)
    decoderR_train_value = pickle.load(fp)
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

pca = PCA(n_components=2)
pdb.set_trace()
pca.fit(encoderR_train_value)
pca_point1 = pca.transform(encoderR_train_value)

x1=pca_point1[:,0]
y1=pca_point1[:,1]

plt.scatter(x1, y1)
plt.show()


