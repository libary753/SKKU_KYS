# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import datetime


class FashionNet_2nd:
    def __init__(self):
        self.RGB_MEAN = np.array([[ 102.9801, 115.9465, 122.7717]],dtype=np.float32)
        self.conv_4 = tf.placeholder(tf.float32, [None, 28, 28, 512])
        self.landmark_visibility = tf.placeholder(tf.float32,[None,8])
        self.landmark_1= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_2= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_3= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_4= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_5= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_6= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_7= tf.placeholder(tf.float32, [None, 3, 3, 512])
        self.landmark_8= tf.placeholder(tf.float32, [None, 3, 3, 512])
        
        self.param=[]
        self.out=[]
    
    
    def conv_layer(self,kernel_shape,bias_shape,bottom,convSt,convB,conv_name,Trainable=True):
        kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype=tf.float32,stddev=convSt),name=conv_name+'_w')
        bias = tf.Variable(tf.constant(convB, shape=bias_shape, dtype=tf.float32),trainable=Trainable,name=conv_name+'_b')
        conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME',name=conv_name)
        out = tf.nn.bias_add(conv, bias,name=conv_name)
        
        self.param +=[kernel,bias]

        return tf.nn.relu(out)
    
        
    def fc_layer(self,bottom,input_shape,output_shape,weight_st,bias,fc_name,relu=True,dropout=False):
        weight = tf.Variable(tf.truncated_normal([input_shape, output_shape],dtype=tf.float32,stddev=weight_st),name=fc_name+'_w')
        bias = tf.Variable(tf.constant(bias, shape=[output_shape], dtype=tf.float32),trainable=True,name=fc_name+'_b')

        self.param +=[weight,bias]

        if relu:
            out = tf.nn.relu(tf.nn.bias_add(tf.matmul(bottom, weight), bias,name=fc_name))
        else:
            out = tf.nn.bias_add(tf.matmul(bottom, weight), bias,name=fc_name)
                
        if dropout:
            return tf.nn.dropout(out,self.keep_prob,name=fc_name)
        else:
            return out
    
            
    def get_roi(self,landmark_x,landmark_y,conv_4,batSize):
        self.landmark_roi=np.zeros((batSize,8,3,3,512),dtype=np.float32)
        for i in range(batSize):
            roi_concat = np.zeros((8,3,3,512),dtype=np.float32)
            for j in range(8):
                x=int((landmark_x[i][j]+0.5)*224)
                y=int((landmark_y[i][j]+0.5)*224)
                
                x=max(1,x)
                x=min(222,x)
                y=max(1,x)
                y=min(222,x)
                
                roi = np.zeros((3,3,512),dtype=np.float32)
                roi = conv_4[i,y:y+3,x:x+3]
                roi_concat[j]=roi
            self.landmark_roi[i]=roi_concat
        
    """
    CNN
    """
    
    def build_net(self,Dropout=False):
        
        """
        keep_prob for dropout
        """
        
        self.keep_prob = tf.placeholder(tf.float32)
        convB = 0.0
        convSt = 0.01
        fcB = 0.0
        fcSt = 0.005
        
        """
        conv_5 global
        """
        self.pool_4 = tf.nn.max_pool(self.conv_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_4')
        self.conv_5_1 = self.conv_layer([3,3,512,512],[512],self.pool_4,convSt,convB,'conv_5_1')
        self.conv_5_2 = self.conv_layer([3,3,512,512],[512],self.conv_5_1,convSt,convB,'conv_5_2')
        self.conv_5_3 = self.conv_layer([3,3,512,512],[512],self.conv_5_2,convSt,convB,'conv_5_3')
        self.pool_global = tf.nn.max_pool(self.conv_5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_global')
          
        """
        pool_landmark
        """
        
        self.pool_landmark_1 = tf.nn.max_pool(self.landmark_1,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_1')
        self.pool_landmark_2 = tf.nn.max_pool(self.landmark_2,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_2')
        self.pool_landmark_3 = tf.nn.max_pool(self.landmark_3,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_3')
        self.pool_landmark_4 = tf.nn.max_pool(self.landmark_4,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_4')
        self.pool_landmark_5 = tf.nn.max_pool(self.landmark_5,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_5')
        self.pool_landmark_6 = tf.nn.max_pool(self.landmark_6,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_6')
        self.pool_landmark_7 = tf.nn.max_pool(self.landmark_7,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_7')
        self.pool_landmark_8 = tf.nn.max_pool(self.landmark_8,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_8')
        
        self.pool_landmark = tf.concat([self.pool_landmark_1,self.pool_landmark_2,self.pool_landmark_3,self.pool_landmark_4,self.pool_landmark_5,self.pool_landmark_6,self.pool_landmark_7,self.pool_landmark_8],0)
        
        
        """
        fc Layer
        """
        
        shape_landmark = int(np.prod(self.pool_landmark.get_shape()[1:]))
        shape_global = int(np.prod(self.pool_global.get_shape()[1:]))
        
        self.pool_landmark_flat = tf.reshape(self.pool_landmark, [-1, shape_landmark])        
        self.pool_global_flat = tf.reshape(self.pool_global, [-1, shape_global])
        
        self.fc_1_landmark=self.fc_layer(self.pool_landmark_flat,shape_landmark,1024,fcSt,fcB,'fc_1_landmark',dropout=Dropout)
        self.fc_1_global=self.fc_layer(self.pool_global_flat,shape_global,4096,fcSt,fcB,'fc_1_global',dropout=Dropout)
        self.fc_1=tf.concat([self.fc_1_landmark,self.fc_1_global],1)
        self.fc_2 = self.fc_layer(self.fc_1,5120,4096,fcSt,fcB,'fc_2',dropout=Dropout)
        self.fc_3_category =self.fc_layer(self.fc_2,4096,15,fcSt,fcB,'out_visibility_2',relu=False)
        #self.fc_3_attribute =self.fc_layer(self.fc_2,4096,1000,fcSt,fcB,'out_visibility_3',relu=False)
        self.out_triplet =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_1',relu=False)
        self.out_category_prob=tf.nn.softmax(self.fc_3_category)
        self.out_visibility_3=tf.nn.softmax(self.fc_3_attribute)


        
    
    #save model
    def save_model(self,sess,path):
        self.saver = tf.train.Saver(self.param)
        self.saver.save(sess,path)           

    #restore model
    #fn.restore_model(sess,'C:/Users/libar/Desktop/save_full/init/model')
    def restore_model(self,sess,path):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(self.param)
        saver.restore(sess,path)
        
    #restore vgg model
    #fn.restore_vgg(sess,'C:/Users/libar/Desktop/save_full/vgg16/model')
    def restore_vgg(self,sess,path):
        sess.run(tf.global_variables_initializer())
        param_vgg=self.param[:26]
        saver = tf.train.Saver(param_vgg)
        saver.restore(sess,path)