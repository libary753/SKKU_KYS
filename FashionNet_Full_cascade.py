# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import datetime


class FashionNet:
    def __init__(self):
        self.RGB_MEAN = np.array([[ 102.9801, 115.9465, 122.7717]],dtype=np.float32)
        self.img = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.param=[]
        self.out=[]
    
    
    def conv_layer(self,kernel_shape,bias_shape,bottom,convSt,convB,conv_name,Trainable=True):
        kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype=tf.float32,stddev=convSt))
        bias = tf.Variable(tf.constant(convB, shape=bias_shape, dtype=tf.float32),trainable=Trainable)
        conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME',name=conv_name)
        out = tf.nn.bias_add(conv, bias,name=conv_name)
        
        self.param +=[kernel,bias]

        return tf.nn.relu(out)
    
        
    def fc_layer(self,bottom,input_shape,output_shape,weight_st,bias,fc_name,relu=True,dropout=False):
        weight = tf.Variable(tf.truncated_normal([input_shape, output_shape],dtype=tf.float32,stddev=weight_st))
        bias = tf.Variable(tf.constant(bias, shape=[output_shape], dtype=tf.float32),trainable=True)

        self.param +=[weight,bias]

        if relu:
            out = tf.nn.relu(tf.nn.bias_add(tf.matmul(bottom, weight), bias,name=fc_name))
        else:
            out = tf.nn.bias_add(tf.matmul(bottom, weight), bias,name=fc_name)
                
        if dropout:
            return tf.nn.dropout(out,self.keep_prob,name=fc_name)
        else:
            return out
            
    """
    CNN
    """
    
    def build_net(self,Dropout=False):
        
        """
        keep_prob for dropout
        """
        
        self.keep_prob = tf.placeholder(tf.float32)
        convB = 0.0
        convSt = 0.1
        fcB = 0.0
        fcSt = 0.1
        
        """
        conv Layer
        """
        
        self.conv_1_1 = self.conv_layer([3,3,3,64],[64],self.img,convSt,convB,'conv_1_1')
        self.conv_1_2 = self.conv_layer([3,3,64,64],[64],self.conv_1_1,convSt,convB,'conv_1_2')
        self.pool_1 = tf.nn.max_pool(self.conv_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
        
        
        self.conv_2_1 = self.conv_layer([3,3,64,128],[128],self.pool_1,convSt,convB,'conv_2_1')
        self.conv_2_2 = self.conv_layer([3,3,128,128],[128],self.conv_2_1,convSt,convB,'conv_2_2')
        self.pool_2 = tf.nn.max_pool(self.conv_2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
                
        self.conv_3_1 = self.conv_layer([3,3,128,256],[256],self.pool_2,convSt,convB,'conv_3_1')
        self.conv_3_2 = self.conv_layer([3,3,256,256],[256],self.conv_3_1,convSt,convB,'conv_3_2')
        self.conv_3_3 = self.conv_layer([3,3,256,256],[256],self.conv_3_2,convSt,convB,'conv_3_3')
        self.pool_3 = tf.nn.max_pool(self.conv_3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
        
        self.conv_4_1 = self.conv_layer([3,3,256,512],[512],self.pool_3,convSt,convB,'conv_4_1')
        self.conv_4_2 = self.conv_layer([3,3,512,512],[512],self.conv_4_1,convSt,convB,'conv_4_2')
        self.conv_4_3 = self.conv_layer([3,3,512,512],[512],self.conv_4_2,convSt,convB,'conv_4_3')
        self.pool_4 = tf.nn.max_pool(self.conv_4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_4')
        
        self.conv_5_1_pose = self.conv_layer([3,3,512,512],[512],self.pool_4,convSt,convB,'conv_5_1')
        self.conv_5_2_pose = self.conv_layer([3,3,512,512],[512],self.conv_5_1_pose,convSt,convB,'conv_5_2')
        self.conv_5_3_pose = self.conv_layer([3,3,512,512],[512],self.conv_5_2_pose,convSt,convB,'conv_5_3')
        self.pool_5_pose = tf.nn.max_pool(self.conv_5_3_pose, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_5')
            
        """
        fc Layer
        """
        
        shape = int(np.prod(self.pool_5_pose.get_shape()[1:]))
        self.pool_5_pose_flat = tf.reshape(self.pool_5_pose, [-1, shape])        
        self.fc_1=self.fc_layer(self.pool_5_pose_flat,shape,4096,fcSt,fcB,'fc_1',dropout=Dropout)
        self.fc_2 = self.fc_layer(self.fc_1,4096,4096,fcSt,fcB,'fc_2',dropout=Dropout) 
        self.fc_3_softlabel=self.fc_layer(self.fc_2,4096,20,fcSt,fcB,'fc_3_pose_softlabel')
        
        self.out_softlabel = tf.nn.softmax(self.fc_3_softlabel,name='out_softlabel')
        self.out_landmark =self.fc_layer(self.fc_2,4096,16,fcSt,fcB,'out_landmark',relu=False)
        self.out_visibility_1 =self.fc_layer(self.fc_2,4096,16,fcSt,fcB,'out_visibility_1',relu=False)
        self.out_visibility_2 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_2',relu=False)
        self.out_visibility_3 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_3',relu=False)
        self.out_visibility_4 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_4',relu=False)
        self.out_visibility_5 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_5',relu=False)
        self.out_visibility_6 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_6',relu=False)
        self.out_visibility_7 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_7',relu=False)
        self.out_visibility_8 =self.fc_layer(self.fc_2,4096,3,fcSt,fcB,'out_visibility_8',relu=False)

        self.out = tf.concat([self.out_softlabel,self.out_landmark,self.out_visibility_1,self.out_visibility_2,self.out_visibility_3,self.out_visibility_4,self.out_visibility_5,self.out_visibility_6,self.out_visibility_7,self.out_visibility_8],1)
                

    #restore model
    def restore_model(self,path):
        saver = tf.train.Saver(self.param)
        saver.restore(self.sess, path)