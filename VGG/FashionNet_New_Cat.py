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
            self.pool_landmark = tf.placeholder(tf.float32, [None, 8, 3, 3, 512],name='pool_landmark')
        elif self.model_type is 'upper':
            self.numOfPoint = 6
            self.pool_landmark = tf.placeholder(tf.float32, [None, 6, 3, 3, 512],name='pool_landmark')
        elif model_type is 'lower':
            self.numOfPoint = 4
            self.pool_landmark = tf.placeholder(tf.float32, [None, 4, 3, 3, 512],name='pool_landmark')
            
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
    
            
    def get_roi(self,l_x,l_y,l_v,conv_4,batSize,model_type,sess):
        
        if model_type is 'full':
            
            roi=np.zeros((batSize,8,3,3,512),dtype=np.float32)
        
            for i in range(batSize):
                self.roi_1 = self.roi_pooling_2point(l_x[i][0],l_x[i][1],l_y[i][0],l_y[i][1],l_v[i][0],l_v[i][1],conv_4,i)
                self.roi_2 = self.roi_pooling_2point(l_x[i][0],l_x[i][2],l_y[i][0],l_y[i][2],l_v[i][0],l_v[i][2],conv_4,i)
                self.roi_3 = self.roi_pooling_2point(l_x[i][1],l_x[i][3],l_y[i][1],l_y[i][3],l_v[i][1],l_v[i][3],conv_4,i)
                self.roi_4 = self.roi_pooling_2point(l_x[i][4],l_x[i][5],l_y[i][4],l_y[i][5],l_v[i][4],l_v[i][5],conv_4,i)
                self.roi_5 = self.roi_pooling_2point(l_x[i][6],l_x[i][7],l_y[i][6],l_y[i][7],l_v[i][6],l_v[i][7],conv_4,i)
                self.roi_6 = self.roi_pooling_2point(l_x[i][4],l_x[i][7],l_y[i][4],l_y[i][7],l_v[i][4],l_v[i][7],conv_4,i)
                self.roi_7 = self.roi_pooling_2point(l_x[i][5],l_x[i][6],l_y[i][5],l_y[i][6],l_v[i][5],l_v[i][6],conv_4,i)
                self.roi_8 = self.roi_pooling_4point(l_x[i][0],l_x[i][1],l_x[i][4],l_x[i][5],l_y[i][0],l_y[i][1],l_y[i][4],l_y[i][5],l_v[i][0],l_v[i][1],l_v[i][4],l_v[i][5],conv_4,i)
                
                roi_concat = np.array([self.roi_1,self.roi_2,self.roi_3,self.roi_4,self.roi_5,self.roi_6,self.roi_7,self.roi_8],dtype=np.float32)
                
                roi[i]=roi_concat

                
        elif model_type is 'upper':
            
            roi=np.zeros((batSize,6,3,3,512),dtype=np.float32)
        
            for i in range(batSize):
                self.roi_1 = self.roi_pooling_2point(l_x[i][0],l_x[i][1],l_y[i][0],l_y[i][1],l_v[i][0],l_v[i][1],conv_4,i)
                self.roi_2 = self.roi_pooling_2point(l_x[i][0],l_x[i][2],l_y[i][0],l_y[i][2],l_v[i][0],l_v[i][2],conv_4,i)
                self.roi_3 = self.roi_pooling_2point(l_x[i][1],l_x[i][3],l_y[i][1],l_y[i][3],l_v[i][1],l_v[i][3],conv_4,i)
                self.roi_4 = self.roi_pooling_2point(l_x[i][4],l_x[i][5],l_y[i][4],l_y[i][5],l_v[i][4],l_v[i][5],conv_4,i)
                self.roi_5 = self.roi_pooling_2point(l_x[i][0],l_x[i][5],l_y[i][0],l_y[i][5],l_v[i][0],l_v[i][5],conv_4,i)
                self.roi_6 = self.roi_pooling_2point(l_x[i][1],l_x[i][4],l_y[i][1],l_y[i][4],l_v[i][1],l_v[i][4],conv_4,i)
                
                roi_concat = np.array([self.roi_1,self.roi_2,self.roi_3,self.roi_4,self.roi_5,self.roi_6],dtype=np.float32)
                
                roi[i]=roi_concat
            
        elif model_type is 'lower':
            roi=np.zeros((batSize,4,3,3,512),dtype=np.float32)
        
            for i in range(batSize):
                self.roi_1 = self.roi_pooling_2point(l_x[i][0],l_x[i][1],l_y[i][0],l_y[i][1],l_v[i][0],l_v[i][1],conv_4,i)
                self.roi_2 = self.roi_pooling_2point(l_x[i][2],l_x[i][3],l_y[i][2],l_y[i][3],l_v[i][2],l_v[i][3],conv_4,i)
                self.roi_3 = self.roi_pooling_2point(l_x[i][0],l_x[i][3],l_y[i][0],l_y[i][3],l_v[i][0],l_v[i][3],conv_4,i)
                self.roi_4 = self.roi_pooling_2point(l_x[i][1],l_x[i][2],l_y[i][1],l_y[i][2],l_v[i][1],l_v[i][2],conv_4,i)
                
                roi_concat = np.array([self.roi_1,self.roi_2,self.roi_3,self.roi_4],dtype=np.float32)
                
                roi[i]=roi_concat
        
        
        return roi
    
    def roi_pooling_2point(self,x1,x2,y1,y2,v1,v2,feature,idx):
        
        #roi 구하기
        
        x1=int((x1+0.5)*28)
        x2=int((x2+0.5)*28)
        y1=int((y1+0.5)*28)
        y2=int((y2+0.5)*28)
        
        x1=max(1,x1)
        x1=min(26,x1)
        x2=max(1,x2)
        x2=min(26,x2)
        y1=max(1,y1)
        y1=min(26,y1)
        y2=max(1,y2)
        y2=min(26,y2)
        
        x_from=min(x1,x2)-1
        x_to=max(x1,x2)+2
        y_from=min(y1,y2)-1
        y_to=max(y1,y2)+2

        x_len=x_to-x_from
        y_len=y_to-y_from
        x_sub=int(x_len/3)
        y_sub=int(y_len/3)
        
        if int(x_len%3) is 0:
            x=[0,x_sub,x_sub*2,x_to]
        elif int(x_len%3) is 1:
            x=[0,x_sub,x_sub*2,x_to]
        elif int(x_len%3) is 2:
            x=[0,x_sub,x_sub*2+1,x_to]
            
        if int(y_len%3) is 0:
            y=[0,y_sub,y_sub*2,y_to]
        elif int(y_len%3) is 1:
            y=[0,y_sub,y_sub*2,y_to]
        elif int(y_len%3) is 2:
            y=[0,y_sub,y_sub*2+1,y_to]
            
        #print('x: ',x)
        #print('y: ',y)
        if int(v1) is 0 and int(v2) is 0:
            v=1
        else:
            v=0
        
        roi = feature[idx,x_from:x_to,y_from:y_to]*v
        
        roi_pooling = np.zeros((3,3,512),dtype=np.float32)
        
        for i in range(3):
            for j in range(3):
                self.sub_feature = roi[x[i]:x[i+1],y[i]:y[i+1],:]
                #print(self.sub_feature.shape)
                for k in range(512):
                    roi_pooling[i][j][k] = self.sub_feature[:,:,k].max()
         
        return roi_pooling
    
    
    
    def roi_pooling_4point(self,x1,x2,x3,x4,y1,y2,y3,y4,v1,v2,v3,v4,feature,idx):
        
        #여백
        x1=int((x1+0.5)*28)
        x2=int((x2+0.5)*28)
        x3=int((x1+0.5)*28)
        x4=int((x2+0.5)*28)
        
        y1=int((y1+0.5)*28)
        y2=int((y2+0.5)*28)
        y3=int((y1+0.5)*28)
        y4=int((y2+0.5)*28)
        
        x1=max(1,x1)
        x1=min(26,x1)
        x2=max(1,x2)
        x2=min(26,x2)
        x3=max(1,x1)
        x3=min(26,x1)
        x4=max(1,x2)
        x4=min(26,x2)
        y1=max(1,y1)
        y1=min(26,y1)
        y2=max(1,y2)
        y2=min(26,y2)
        y3=max(1,y1)
        y3=min(26,y1)
        y4=max(1,y2)
        y4=min(26,y2)
        
        x_from=min(x1,x2,x3,x4)-1
        x_to=max(x1,x2,x3,x4)+2
        y_from=min(y1,y2,y3,y4)-1
        y_to=max(y1,y2,y3,y4)+2
        
        x_len=x_to-x_from
        y_len=y_to-y_from
        x_sub=int(x_len/3)
        y_sub=int(y_len/3)


        if int(x_len%3) is 0:
            x=[0,x_sub,x_sub*2,x_to]
        elif int(x_len%3) is 1:
            x=[0,x_sub,x_sub*2,x_to]
        elif int(x_len%3) is 2:
            x=[0,x_sub,x_sub*2+1,x_to]
            
        if int(y_len%3) is 0:
            y=[0,y_sub,y_sub*2,y_to]
        elif int(y_len%3) is 1:
            y=[0,y_sub,y_sub*2,y_to]
        elif int(y_len%3) is 2:
            y=[0,y_sub,y_sub*2+1,y_to]
        
        #print('x: ',x)
        #print('y: ',y)
        
        if int(v1) is 0 and int(v2) is 0:
            v=1
        else:
            v=0
        
        roi = feature[idx,x_from:x_to,y_from:y_to]*v
        
        roi_pooling = np.zeros((3,3,512),dtype=np.float32)
        for i in range(3):
            for j in range(3):
                self.sub_feature = roi[x[i]:x[i+1],y[i]:y[i+1],:]
                #print(sub_feature.shape)
                for k in range(512):
                    roi_pooling[i][j][k] = self.sub_feature[:,:,k].max()
    
        return roi_pooling
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