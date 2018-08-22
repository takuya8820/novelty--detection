# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:04:56 2018

@author: takuya
"""

import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np
import math, os
import pickle
import pdb
import input_data
import matplotlib.pylab as plt

# get_variableは既に存在あれば取得なし、無ければ変数を作成
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=0.1))
    
def bias_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

#padding='SAME' はゼロパティングしていること
# inputsで4次元のテンソルを返す   
def conv1d_relu(inputs, w, b, stride):
    conv = tf.nn.conv1d(inputs, w, stride, padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv

def conv1d_t_relu(inputs, w, b, output_shape, stride):
    conv = nn_ops.conv1d_transpose(inputs, w, output_shape=output_shape, stride=stride, padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv
    
def conv2d_relu(inputs, w, b, stride):    
    #tf.nn.conv2d(input,filter,strides,padding)
    #input 4次元([batch, in_height, in_width, in_channels])のテンソルを渡す
    #filter 畳込みでinputテンソルとの積和に使用するweightにあたる
    #stride （=１画素ずつではなく、数画素ずつフィルタの適用範囲を計算するための値)を指定.ただし指定は[1, stride, stride, 1]と先頭と最後は１固定とする
    conv = tf.nn.conv2d(inputs, w, strides=stride, padding='SAME') + b 
    conv = tf.nn.relu(conv)
    return conv

def conv2d_t_relu(inputs, w, b, output_shape, stride):
    conv = tf.nn.conv2d_transpose(inputs, w, output_shape=output_shape, strides=stride, padding='SAME') + b
    conv = tf.nn.relu(conv)
    return conv
    
def fc_relu(inputs, w, b):
    fc = tf.matmul(inputs, w) + b
    fc = tf.nn.relu(fc)
    return fc

def fc_sigmoid(inputs, w,b):
    return 1/(1 + np.exp(-inputs))
#----------------
    

#reuse=Trueで再利用できるように
#tf.variable_scope() は，変数の管理に用いるスコープ定義
def encoderImg(x, z_dim, reuse=None):
    with tf.variable_scope('encoderImg') as scope:
        if reuse:
            scope.reuse_variables()
    
        convW1 = weight_variable("convW1", [3, 3, 1, 32])
        convB1 = bias_variable("convB1", [32])
        conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
 
        convW2 = weight_variable("convW2", [3, 3, 32, 32])
        convB2 = bias_variable("convB2", [32])
        conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
        
        # 2次元画像を１次元に変更して全結合層へ渡す
        # np.prod で配列要素の積を算出
        # get_shape()でサイズを取得する場合TypeはTensorShapeになっている。
        # tf.reshapeの-1は特にサイズを指定しないということ
        
        conv2size = np.prod(conv2.get_shape().as_list()[1:])
        conv2 = tf.reshape(conv2, [-1, conv2size])
       
        fcW1 = weight_variable("fcW1", [conv2size,z_dim])
        fcB1 = bias_variable("fcB1", [z_dim])
        fc1 = fc_relu(conv2, fcW1, fcB1)
        
        return fc1
    
def decoderImg(z,z_dim,reuse=None):
    # tf.variable_scope()は，変数（識別子）を管理するための専用のスコープ定義
    with tf.variable_scope('decoderImg') as scope:
        if reuse:
            scope.reuse_variables()

        fcW1 = weight_variable("fcW1", [z_dim, 7*7*32])
        fcB1 = bias_variable("fcB1", [7*7*32])
        fc1 = fc_relu(z, fcW1, fcB1)
        
        batch_size = tf.shape(fc1)[0]
        # tf.stackはtensorのランクを+1する効果がある
        fc1 = tf.reshape(fc1, tf.stack([batch_size, 7, 7, 32]))
        
        # when padding='SAME', O = I*S
        convW1 = weight_variable("convW1", [3, 3, 32, 32])
        convB1 = bias_variable("convB1", [32])
        conv1 = conv2d_t_relu(fc1, convW1, convB1, output_shape=[batch_size,14,14,32], stride=[1,2,2,1])
        
        convW2 = weight_variable("convW2", [3, 3, 1, 32])
        convB2 = bias_variable("convB2", [1])
        output = conv2d_t_relu(conv1, convW2, convB2, output_shape=[batch_size,28,28,1], stride=[1,2,2,1])
        

        return output
#Dのネットワーク
def d_network_architecture(x, z_dim = 1 , reuse=False):
    with tf.variable_scope('d') as scope:
        if reuse:
            scope.reuse_variables()
        convW1 = weight_variable("convW1", [3, 3, 1, 32])
        convB1 = bias_variable("convB1", [32])
        conv1 = conv2d_relu(x, convW1, convB1, stride=[1,2,2,1])
        
        convW2 = weight_variable("convW2", [3, 3, 32, 32])
        convB2 = bias_variable("convB2", [32])
        conv2 = conv2d_relu(conv1, convW2, convB2, stride=[1,2,2,1])
        
        # 2次元画像を１次元に変更して全結合層へ渡す
        # np.prod で配列要素の積を算出
        # get_shape()でサイズを取得する場合TypeはTensorShapeになっている。
        # tf.reshapeの-1は特にサイズを指定しないということ
        conv2size = np.prod(conv2.get_shape().as_list()[1:])
        conv2 = tf.reshape(conv2, [-1, conv2size])
        
        fcW1 = weight_variable("fcW1", [conv2size,z_dim])
        fcB1 = bias_variable("fcB1", [z_dim])
        fc1 = fc_relu(conv2, fcW1, fcB1)
        
        return fc1
        
def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

# load data
myImage = input_data.read_data_sets("MNIST_data/",one_hot=True)
z_img_dim = 100

#　x_img 28×28×1チャンネル×NONE枚
#教師データ
x_img = tf.placeholder(tf.float32, shape=[None,28,28,1])
x_in_img = tf.placeholder(tf.float32, shape=[None,28,28,1])
d_network_label_img = tf.placeholder(tf.float32, shape=[None,1])
noise_x_img = tf.placeholder(tf.float32, shape=[None,28,28,1])
                

# ノイズ版学習用
train_noise_z_img_op = encoderImg(noise_x_img, z_img_dim)
train_noise_xr_img_op = decoderImg(train_noise_z_img_op, z_img_dim)
train_d_concat_img = tf.concat([x_img,train_noise_xr_img_op] ,0)
train_d_concat_img_op = d_network_architecture(train_d_concat_img)
train_d_img_op = d_network_architecture(x_img, reuse = True) #D(X)
train_d_noise_img_op = d_network_architecture(train_noise_xr_img_op, reuse = True) #D(R(X~))


#ノイズ版テスト用
test_noise_z_img_op = encoderImg(noise_x_img, z_img_dim, reuse=True)
test_noise_xr_img_op = decoderImg(test_noise_z_img_op, z_img_dim, reuse=True)
test_d_concat_img = tf.concat([x_img,test_noise_xr_img_op] ,0)
test_d_concat_img_op = d_network_architecture(test_d_concat_img, reuse = True)



log_d = tf.log(train_d_img_op)#/x_img.shape[0]
log_dr = tf.log(1 - train_d_noise_img_op)#/noise_x_img.shape[0]

#　tf.reduce_meanは与えたリストに入っている数値の平均値を求める関数
#　tf.squareは要素ごとに二乗をとる
loss_r = tf.reduce_mean(tf.square(train_noise_xr_img_op - x_img)) #式(4)
loss_rd_img = log_d #+ log_dr #式(3)
lamda = 0.4
loss_r_img = lamda*loss_r

#loss_rの最小値を取得
trainer_img = tf.train.AdamOptimizer(1e-3).minimize(loss_r_img)
#loss_rdの最大値を取得
trainer_1_img = tf.train.AdamOptimizer(1e-3).minimize(loss_rd_img)

batch_size = 200
batch_size_all = 200
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Start training
for i in range(3000):
    batch = myImage.train.next_batch(batch_size) 
    batch_x_img = np.reshape(batch[0], (batch_size,28,28,1))
    #1のみ
    batch_1_img = batch[0][batch[1][:,1]==1]
    batch_x_1_img = np.reshape(batch_1_img, (batch_1_img.shape[0],28,28,1))
    
    #trainのラベル
    train_d_network_label_img = np.zeros([batch_x_1_img.shape[0]*2,1])
    train_d_network_label_img[:batch_x_1_img.shape[0]] = 0.9
    train_d_network_label_img[batch_x_1_img.shape[0]:] = 0.1
    

    #train_d_network_label_img[batch_1_img.shape[0]:,1] = 1
     
    #ノイズを追加する(ガウシアンノイズ)
    sheet,row,column,ch = batch_x_1_img.shape
    mean = 0
    sigma = 0.5
    #np.random.normal(平均、分散、出力する件数)　正規分布に従う乱数を出力
    gauss = np.random.normal(mean,sigma,(batch_size,row,column,ch))
    gauss_1= np.random.normal(mean,sigma,(sheet,row,column,ch)) 
    
    #ノイズ付きデータ
    noise_batch_x_img = batch_x_img + gauss
    noise_batch_x_1_img = batch_x_1_img + gauss_1
    
    _, _1, r_img, rd_img, train_xr_img, train_z_img, train_d_img = sess.run([trainer_img,trainer_1_img, loss_r_img, loss_rd_img, train_noise_xr_img_op, train_noise_z_img_op, train_d_concat_img_op], feed_dict={x_img: batch_x_1_img, noise_x_img: noise_batch_x_1_img, d_network_label_img:train_d_network_label_img})
    if i % 10 == 0:
        #print("Image, iteration: %d, r loss is %f, rd loss is %f" % (i,r_img, rd_img))
        print("Image, iteration: %d, r loss is %f" % (i,r_img))
        
    if i % 50 == 0:
        batch_test = myImage.test.next_batch(batch_size_all)
        batch_x_test_img = np.reshape(batch_test[0],(batch_size_all,28,28,1))
        
        #1のみ
        batch_1_img = batch_test[0][batch_test[1][:,1]==1] 
        batch_test_1_img = np.reshape(batch_1_img,(batch_1_img.shape[0],28,28,1))
        #1以外
        batch_not_1_img = batch_test[0][batch_test[1][:,1]!=1]
        batch_test_not_1_img = np.reshape(batch_not_1_img,(batch_not_1_img.shape[0],28,28,1))
      
        
        number = round(batch_test_1_img.shape[0]*0.3)
        #1の30％程度の1以外データ
        batch_number_not_1_img = batch_test_not_1_img[:number]
        
        #入力データ
        batch_1_test_img = np.append(batch_1_img, batch_number_not_1_img, axis=0)  
        batch_x_1_test_img = np.reshape(batch_1_test_img,(batch_1_test_img.shape[0],28,28,1))
                
        test_d_network_label_img = np.zeros([batch_test_1_img.shape[0]+batch_test_not_1_img.shape[0],1])
        test_d_network_label_img[:batch_test_1_img.shape[0]] = 0.9
        test_d_network_label_img[batch_test_1_img.shape[0]:] = 0.1
        
        #test_d_network_label_img[batch_test_1_img.shape[0]:,1] = 1
        
        noise_batch_x_test_img = batch_x_test_img + gauss
        
        #test_d_img = sess.run([test_d_concat_img_op], feed_dict={x_in_img: batch_x_1_test_img, x_img: batch_test_1_img, noise_x_img: batch_test_not_1_img, d_network_label_img:test_d_network_label_img})

                
        test_xr_img, test_z_img, test_d_img = sess.run([test_noise_xr_img_op, test_noise_z_img_op, test_d_concat_img_op], feed_dict={x_in_img: batch_x_1_test_img, x_img: batch_test_1_img, noise_x_img: batch_test_not_1_img, d_network_label_img:test_d_network_label_img})
        
        #　緑字のファイル名をバイナリ形式(2進数)で保存するために開いて変数fpに代入
        with open("./visualization/img_{}.pickle".format(i), "wb") as fp:
        #　変数fpに対してその前の変数の内容を書き込んでいる
            pickle.dump(noise_batch_x_img,fp)  
            pickle.dump(train_xr_img,fp)
            pickle.dump(train_z_img,fp)
            pickle.dump(batch_x_test_img,fp)
            pickle.dump(noise_batch_x_test_img,fp)
            pickle.dump(test_xr_img,fp)
            pickle.dump(test_z_img,fp)
            pickle.dump(test_d_img,fp)
        
        '''
        fig, figInds = plt.subplots(nrows=2, ncols=10, sharex=True)
    
        for figInd in np.arange(figInds.shape[1]):

            fig0 = figInds[0][figInd].imshow(batch_x_1_test_img[figInd,:,:,0])
            fig1 = figInds[1][figInd].imshow(batch_x_not_1_test_img[figInd,:,:,0])
            #fig2 = figInds[2][figInd].imshow(test_xr_img[figInd,:,:,0])
            
            fig0.axes.get_xaxis().set_visible(False)
            fig0.axes.get_yaxis().set_visible(False)
            fig0.axes.get_xaxis().set_ticks([])
            fig0.axes.get_yaxis().set_ticks([])
            fig1.axes.get_xaxis().set_visible(False)
            fig1.axes.get_yaxis().set_visible(False)
            fig1.axes.get_xaxis().set_ticks([])
            fig1.axes.get_yaxis().set_ticks([])
            #fig2.axes.get_xaxis().set_visible(False)
            #fig2.axes.get_yaxis().set_visible(False)
            #fig2.axes.get_xaxis().set_ticks([])
            #fig2.axes.get_yaxis().set_ticks([])
        
        plt.savefig("./visualization_1/img_{}.png".format(i))
        '''
        # save model to file
        saver = tf.train.Saver()
        saver.save(sess,"./modelsMNIST_CNN_1/img_{}.ckpt".format(i))      