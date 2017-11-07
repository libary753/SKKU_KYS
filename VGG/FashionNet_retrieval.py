# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image


class FashionNet:
    def __init__(self,model_type):
        self.model_type=model_type
        self.RGB_MEAN = np.array([[ 102.9801, 115.9465, 122.7717]],dtype=np.float32)
        self.img = tf.placeholder(tf.float32, [None, 224, 224, 3])
        if self.model_type is 'full':
            self.numOfPoint = 8    
            self.landmark_1= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_1')
            self.landmark_2= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_2')
            self.landmark_3= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_3')
            self.landmark_4= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_4')
            self.landmark_5= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_5')
            self.landmark_6= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_6')
            self.landmark_7= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_7')
            self.landmark_8= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_8')    
        elif self.model_type is 'upper':
            self.numOfPoint = 6
            self.landmark_1= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_1')
            self.landmark_2= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_2')
            self.landmark_3= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_3')
            self.landmark_4= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_4')
            self.landmark_5= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_5')
            self.landmark_6= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_6')
        elif model_type is 'lower':
            self.numOfPoint = 4
            self.landmark_1= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_1')
            self.landmark_2= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_2')
            self.landmark_3= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_3')
            self.landmark_4= tf.placeholder(tf.float32, [None, 3, 3, 512],name='landmark_4')
        
        self.landmark_visibility = tf.placeholder(tf.float32,[None,self.numOfPoint])
        self.param=[]
        self.out=[]
    
    
    def conv_layer(self,kernel_shape,bias_shape,bottom,convSt,convB,conv_name,Trainable=True):
        kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype=tf.float32,stddev=convSt),trainable=Trainable,name=conv_name+'_w')
        bias = tf.Variable(tf.constant(convB, shape=bias_shape, dtype=tf.float32),trainable=Trainable,name=conv_name+'_b')
        conv = tf.nn.conv2d(bottom, kernel, [1, 1, 1, 1], padding='SAME',name=conv_name)
        out = tf.nn.bias_add(conv, bias,name=conv_name)
        
        self.param +=[kernel,bias]

        return tf.nn.relu(out)
    
        
    def fc_layer(self,bottom,input_shape,output_shape,weight_st,bias,fc_name,relu=True,dropout=False,Trainable=True):
        weight = tf.Variable(tf.truncated_normal([input_shape, output_shape],dtype=tf.float32,stddev=weight_st),trainable=Trainable,name=fc_name+'_w')
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
    
            
    def get_roi(self,landmark_x,landmark_y,conv_4,batSize,model_type):
        if model_type is 'full':
            numOfPoint = 8
            
        elif model_type is 'upper':
            numOfPoint = 6
            
        elif model_type is 'lower':
            numOfPoint = 4
        
        
        self.landmark_roi=np.zeros((batSize,numOfPoint,3,3,512),dtype=np.float32)
        for i in range(batSize):
            roi_concat = np.zeros((numOfPoint,3,3,512),dtype=np.float32)
            for j in range(numOfPoint):
                x=int((landmark_x[i][j]+0.5)*28)
                y=int((landmark_y[i][j]+0.5)*28)
                
                x=max(1,x)
                x=min(26,x)
                y=max(1,y)
                y=min(26,y)
                
                roi = np.zeros((3,3,512),dtype=np.float32)
                roi = conv_4[i,x-1:x+2,y-1:y+2]
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
        conv Layer
        """
        
        self.conv_1_1 = self.conv_layer([3,3,3,64],[64],self.img,convSt,convB,'conv_1_1',Trainable=False)
        self.conv_1_2 = self.conv_layer([3,3,64,64],[64],self.conv_1_1,convSt,convB,'conv_1_2',Trainable=False)
        self.pool_1 = tf.nn.max_pool(self.conv_1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
        
        
        self.conv_2_1 = self.conv_layer([3,3,64,128],[128],self.pool_1,convSt,convB,'conv_2_1',Trainable=False)
        self.conv_2_2 = self.conv_layer([3,3,128,128],[128],self.conv_2_1,convSt,convB,'conv_2_2',Trainable=False)
        self.pool_2 = tf.nn.max_pool(self.conv_2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
                
        self.conv_3_1 = self.conv_layer([3,3,128,256],[256],self.pool_2,convSt,convB,'conv_3_1',Trainable=False)
        self.conv_3_2 = self.conv_layer([3,3,256,256],[256],self.conv_3_1,convSt,convB,'conv_3_2',Trainable=False)
        self.conv_3_3 = self.conv_layer([3,3,256,256],[256],self.conv_3_2,convSt,convB,'conv_3_3',Trainable=False)
        self.pool_3 = tf.nn.max_pool(self.conv_3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_3')
        
        self.conv_4_1 = self.conv_layer([3,3,256,512],[512],self.pool_3,convSt,convB,'conv_4_1',Trainable=False)
        self.conv_4_2 = self.conv_layer([3,3,512,512],[512],self.conv_4_1,convSt,convB,'conv_4_2',Trainable=False)
        self.conv_4_3 = self.conv_layer([3,3,512,512],[512],self.conv_4_2,convSt,convB,'conv_4_3',Trainable=False)
        self.pool_4 = tf.nn.max_pool(self.conv_4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_4')
        
        self.conv_5_1 = self.conv_layer([3,3,512,512],[512],self.pool_4,convSt,convB,'conv_5_1')
        self.conv_5_2 = self.conv_layer([3,3,512,512],[512],self.conv_5_1,convSt,convB,'conv_5_2')
        self.conv_5_3 = self.conv_layer([3,3,512,512],[512],self.conv_5_2,convSt,convB,'conv_5_3')
        self.pool_global = tf.nn.max_pool(self.conv_5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_5_global')
          
        """
        pool_landmark
        """
        
        if self.model_type is 'full':
            self.pool_landmark_1 = tf.nn.max_pool(self.landmark_1,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_1')
            self.pool_landmark_2 = tf.nn.max_pool(self.landmark_2,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_2')
            self.pool_landmark_3 = tf.nn.max_pool(self.landmark_3,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_3')
            self.pool_landmark_4 = tf.nn.max_pool(self.landmark_4,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_4')
            self.pool_landmark_5 = tf.nn.max_pool(self.landmark_5,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_5')
            self.pool_landmark_6 = tf.nn.max_pool(self.landmark_6,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_6')
            self.pool_landmark_7 = tf.nn.max_pool(self.landmark_7,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_7')
            self.pool_landmark_8 = tf.nn.max_pool(self.landmark_8,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_8')            
            self.pool_landmark = tf.concat([self.pool_landmark_1,self.pool_landmark_2,self.pool_landmark_3,self.pool_landmark_4,self.pool_landmark_5,self.pool_landmark_6,self.pool_landmark_7,self.pool_landmark_8],1)
        
        
        elif self.model_type is 'upper':
            self.pool_landmark_1 = tf.nn.max_pool(self.landmark_1,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_1')
            self.pool_landmark_2 = tf.nn.max_pool(self.landmark_2,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_2')
            self.pool_landmark_3 = tf.nn.max_pool(self.landmark_3,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_3')
            self.pool_landmark_4 = tf.nn.max_pool(self.landmark_4,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_4')
            self.pool_landmark_5 = tf.nn.max_pool(self.landmark_5,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_5')
            self.pool_landmark_6 = tf.nn.max_pool(self.landmark_6,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_6')
            self.pool_landmark = tf.concat([self.pool_landmark_1,self.pool_landmark_2,self.pool_landmark_3,self.pool_landmark_4,self.pool_landmark_5,self.pool_landmark_6],1)
        
        
        elif self.model_type is 'lower':
            self.pool_landmark_1 = tf.nn.max_pool(self.landmark_1,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_1')
            self.pool_landmark_2 = tf.nn.max_pool(self.landmark_2,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_2')
            self.pool_landmark_3 = tf.nn.max_pool(self.landmark_3,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_3')
            self.pool_landmark_4 = tf.nn.max_pool(self.landmark_4,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME',name='pool_landmark_4')
            self.pool_landmark = tf.concat([self.pool_landmark_1,self.pool_landmark_2,self.pool_landmark_3,self.pool_landmark_4],1)
        

        
        
        
        """
        fc Layer
        """
        
        shape_landmark = int(np.prod(self.pool_landmark.get_shape()[1:]))
        shape_global = int(np.prod(self.pool_global.get_shape()[1:]))
        
        self.pool_landmark_flat = tf.reshape(self.pool_landmark, [-1, shape_landmark])        
        self.pool_global_flat = tf.reshape(self.pool_global, [-1, shape_global])
        
        self.fc_1_landmark=self.fc_layer(self.pool_landmark_flat,shape_landmark,1024,fcSt,fcB,'fc_1_landmark',dropout=Dropout)
        self.fc_1_global=self.fc_layer(self.pool_global_flat,shape_global,4096,fcSt,fcB,'fc_1_global',dropout=Dropout)

        self.fc_1=tf.concat([tf.nn.l2_normalize(self.fc_1_landmark,1),tf.nn.l2_normalize(self.fc_1_global,1)],1)
        #self.fc_2 = self.fc_layer(self.fc_1,5120,4096,fcSt,fcB,'fc_2',dropout=Dropout,relu=False)
        self.fc_2 = tf.nn.l2_normalize(self.fc_layer(self.fc_1,5120,4096,fcSt,fcB,'fc_2',dropout=Dropout),1)
        
        if self.model_type is 'full':
            self.fc_3_category = self.fc_layer(self.fc_2,4096,6,fcSt,fcB,'fc_3_category',relu=False)
            
        elif self.model_type is 'upper':
            self.fc_3_category = self.fc_layer(self.fc_2,4096,17,fcSt,fcB,'fc_3_category',relu=False)
            
        elif self.model_type is 'lower':
            self.fc_3_category = self.fc_layer(self.fc_2,4096,12,fcSt,fcB,'fc_3_category',relu=False)
        
        self.cat_prob=tf.nn.softmax(self.fc_3_category)        
                   
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