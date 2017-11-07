# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 00:57:00 2017

@author: libar
"""

# -*- coding: utf-8 -*-

import csv
import tensorflow as tf 
import numpy as np
from PIL import Image
import FashionNet_Attribute_Prediction_for_training as FashionNet
import random

class trainer:
    def __init__(self):
        self.RGB_MEAN = np.array([[ 103.939, 116.779, 123.68 ]],dtype=np.float32)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.param=[]
        self.out=[]
        self.landList=[]
        self.imgList=[]
        self.catList=[]
        self.attrList=[]
        
    """
    image 로딩, 리사이징
    output 로딩
    
    img=test.load_image('C:/Users/libar/Desktop/Landmark Prediction/img/img_00000001.jpg')
    """
    
    def load_batch_for_attribute(self,batNum,batSize=20):
        self.batSize=batSize
        self.batch = np.zeros((batSize,224,224,3),dtype=np.float32) # batch 초기화
    
        imgDir=('C:/Users/libar/Desktop/Attribute Prediction/')
        self.setOutput_attribute(batNum,batSize)
        
        for i in range(batSize):
            imageFileName = imgDir+self.imgList[batNum*batSize+i]
            img = Image.open(imageFileName)
            self.landmark_x[i]=self.landmark_x[i]/img.size[0]-0.5
            self.landmark_y[i]=self.landmark_y[i]/img.size[1]-0.5
            img = self.norm_image(img)
            self.batch[i]=img
            
        self.find_pn(batSize)
    
    def load_image(self,path):
        self.batch = np.zeros((1,224,224,3),dtype=np.float32) # batch 초기화
    
        img = Image.open(path)
        self.batch[0]=self.norm_image(img)
    
    def norm_image(self,img):
        RGB_MEAN = np.array([[ 103.939, 116.779, 123.68 ]],dtype=np.float32)
        scale = 224/max(img.size)
        s1=round(img.size[0]*scale)
        s2=round(img.size[1]*scale)
        nump=np.zeros((224,224,3),dtype=np.float32)
        
        #image size가 224이면 RGB MEAN만 빼줌
        if img.size[0] is 224 and img.size[1] is 224:
            nump=img-RGB_MEAN
        #image size가 224가 아니면 size에 맞춰서 크기 변경
        else:
            img = img.resize((s1,s2))
            img_resized = Image.new('RGB',(224,224))
            if img.size[0] is 224:
                offset=round((224-img.size[1])/2)
                img_resized.paste(img,(0,offset))      
                nump=img_resized-RGB_MEAN
            else:
                offset=round((224-img.size[0])/2)
                img_resized.paste(img,(offset,0))        
                nump=img_resized-RGB_MEAN
            #img_resized.show()
        nump=np.swapaxes(nump,0,1)
        nump=nump[:,:,(2,1,0)]
        return nump
    
    """
    csv에서 파일 정보 읽어오기
    """
    
    def readCsv_attribute(self):
        self.landList=[]
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Land_'+self.model_type+'.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.landList.append(i)
        f.close()
        
        random.shuffle(self.landList)
        
        arr=np.array(self.landList)
        
        for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if len(arr[i][j]) is 0:
                        arr[i][j]='0'
        
        if self.model_type is 'full':
            self.visibility_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'),arr[:,23].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'),arr[:,24].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'),arr[:,22].astype('float32'),arr[:,25].astype('float32'))).T
            
        elif self.model_type is 'upper':
            self.visibility_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'))).T
            
        elif self.model_type is 'lower':
            self.visibility_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'))).T

        self.catList = []
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Cat.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:
            self.catList+=[int(i[0])]
        f.close()

        self.imgList = []
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Img.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:
            self.imgList+=[i[0]]
        f.close()

        self.attrList = []
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Attr.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:
            self.attrList+=[i[0]]
        f.close()
            
    def setOutput_attribute(self,batNum,batSize):
        if self.model_type is 'full':
            self.cat_prob=np.zeros((batSize,10))
            
        elif self.model_type is 'upper':
            self.cat_prob=np.zeros((batSize,20))
            
        elif self.model_type is 'lower':
            self.cat_prob=np.zeros((batSize,16))
            
        self.cat_output=np.zeros((batSize))
        self.attr_prob=np.zeros((batSize,1000,2))
        self.attr_output=np.zeros((batSize,1000))
        self.img_name=[]
        
        
        
        self.landmark_v=self.visibility_list[batSize*batNum:batSize*(batNum+1),:]%2
        self.landmark_x=self.x_list[batSize*batNum:batSize*(batNum+1),:] 
        self.landmark_y=self.y_list[batSize*batNum:batSize*(batNum+1),:]
        
        for i in range(batSize):
            self.cat_output[i] = self.catList[int(tr.landList[batSize*batNum+i][0])]
            
            if self.model_type is 'full':
                index = int(self.cat_output[i])-37
                if(index>0): index = index-1
                if(index>6): index = index-1
                
                self.cat_prob[i][index]=1
                
            elif self.model_type is 'upper':
                self.cat_prob[i][int(self.cat_output[i])-21]=1
                
            elif self.model_type is 'lower':
                self.cat_prob[i][int(self.cat_output[i])-1]=1
            
            self.attr_output[i]=list(map(int,tr.attrList[int(tr.landList[batSize*batNum+i][0])].split(' ')))
            self.img_name+=[self.imgList[int(tr.landList[batSize*batNum+i][0])]]
            for j in range(1000):
                self.attr_prob[i][j][int(((self.attr_output[i][j])+1)/2)]=1
                
         
        self.output=[self.cat_output,self.cat_prob,self.attr_prob]
        
        """
        self.landmark_v=self.visibility_list[batSize*batNum:batSize*(batNum+1),:]%2
        self.landmark_x=self.x_list[batSize*batNum:batSize*(batNum+1),:] 
        self.landmark_y=self.y_list[batSize*batNum:batSize*(batNum+1),:]
        self.output=[category_output,category_prob]
        self.img_name=self.imgList[batSize*batNum:batSize*(batNum+1)]
        """
    def define_loss_attribute(self,fn,batSize=20,margin=0.01):
        #category cross entropy loss
        
        if self.model_type is 'full':
            self.output_category = tf.placeholder(tf.float32, [batSize, 10])
            
        elif self.model_type is 'upper':
            self.output_category = tf.placeholder(tf.float32, [batSize, 20])
            
        elif self.model_type is 'lower':
            self.output_category = tf.placeholder(tf.float32, [batSize, 16])
            

        #self.loss_category = tf.losses.softmax_cross_entropy(self.output_category,fn.out_category_prob)
        self.loss_category = tf.reduce_mean(-tf.reduce_sum(tf.matmul(tr.output_category * tf.log(fn.out_category_prob),tr.cat_weight.T), reduction_indices=1))
        
        #attribute cross entropy loss
        self.output_attribute = tf.placeholder(tf.float32, [batSize, 1000, 2])
        #self.loss_attribute = tf.losses.softmax_cross_entropy(self.output_attribute,fn.out_attribute_prob)
        
        #weighted cross entropy
        
        self.loss_attribute =  tf.reduce_mean(-tf.reduce_sum(self.output_attribute * tf.log(fn.out_attribute_prob), reduction_indices=1))
        
        #triplet loss
        self.pos = tf.placeholder(tf.int32, [batSize])
        self.neg = tf.placeholder(tf.int32, [batSize])

        feature_pos = tf.gather(fn.fc_2,self.pos)
        feature_neg = tf.gather(fn.fc_2,self.neg)
        
        self.p = self.sqrt_dist(fn.fc_2, feature_pos)
        self.n = self.sqrt_dist(fn.fc_2, feature_neg)
        
        self.loss_triplet = tf.reduce_sum(tf.maximum(0.,self.p-self.n+margin))
        
        w_cat = 0.1
        w_att = 0
        w_tri = 1
        
        self.loss = w_cat * self.loss_category + w_att * self.loss_attribute + w_tri * self.loss_triplet 

                    
    def sqrt_dist(self,p1,p2): 
        return tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p1, p2)),1))
    
    def dist(self,p1,p2):
        return np.sqrt(np.sum(np.square(p1-p2)))
    
    def find_pn(self,batsize):
        self.dist_map = np.zeros((batsize,batsize),dtype=np.float32)
        
        for i in range(batsize):
            for j in range(batsize):
                self.dist_map[i][j] = self.dist(self.attr_prob[i],self.attr_prob[j])
                if self.cat_output[i] is self.cat_output[j]:
                    self.dist_map[i][j] = self.dist_map[i][j]-5 #category가 같으면 거리에서 5을 빼줌
                if i is j:
                    self.dist_map[i][j] = 0
        
        self.neg_feed=np.argmax(self.dist_map,0)
        
        for i in range(batsize):
            self.dist_map[i][i] = np.finfo(np.float32).max
        
        self.pos_feed=np.argmin(self.dist_map,0)
        
        
        
    def train_attribute(self):
        self.model_type='full'
        self.readCsv_attribute()
        fn=FashionNet.FashionNet_2nd()
        fn.build_net(model_type=self.model_type,Dropout=True)
        batsize=20
        self.defineWeight(batsize)
        self.define_loss_attribute(fn,batSize=batsize)
        #learningRate= 0.0001(3 epoch까지) 0.00001(그 다음 5 epoch)
        learningRate = 0.000001
        
        train = tf.train.AdamOptimizer(learningRate).minimize(self.loss)
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        #fn.restore_model(sess,'C:/Users/libar/Desktop/cat_full/init/model') 
        fn.restore_model(sess,'C:/Users/libar/Desktop/cat_full/4 epoch/final/model') 
        print('--------------------------------------------------------------')                
                
        for j in range(5,15):
            self.readCsv_attribute()
            for i in range(2620):
                self.load_batch_for_attribute(i,batsize)
                conv_4=sess.run(fn.conv_4_3,feed_dict={fn.imgs:self.batch})
                fn.get_roi(self.landmark_x,self.landmark_y,conv_4,batsize)
                sess.run(train,feed_dict={fn.imgs:tr.batch,fn.landmark_visibility:tr.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],tr.output_category:tr.cat_prob,tr.output_attribute:tr.attr_prob,tr.pos:tr.pos_feed,tr.neg:tr.neg_feed,fn.keep_prob:0.5})
                if i%50 is 0:
                    print('< ',str(j),' epoch, ',str(i),'번째 batch >')
                    [triplet_loss,cat,cat_loss,att_loss]=sess.run([tr.loss_triplet,fn.out_category_prob,self.loss_category,self.loss_attribute],feed_dict={fn.imgs:tr.batch,fn.landmark_visibility:tr.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],tr.output_category:tr.cat_prob,tr.output_attribute:tr.attr_prob,tr.pos:tr.pos_feed,tr.neg:tr.neg_feed,fn.keep_prob:0.5})
                    print('triplet_loss: ',triplet_loss)
                    print('cat_loss: ',cat_loss)
                    print('att_loss: ',att_loss)
                    print('cat: ',np.argmax(cat,1))
                    print('ground_truth: ',np.array(self.cat_output,dtype=np.int32))
                    print('--------------------------------------------------------------')
                    if i%400 is 0:
                        fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(j)+' epoch/'+str(i)+'/model')
                        
            fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(j)+' epoch/final/model')

tr = trainer();        
tr.train_attribute();
"""
fn=FashionNet.FashionNet_2nd()
tr = trainer()
fn.build_net('full')
tr.model_type='full'
tr.readCsv_attribute()
batsize=20
tr.batSize=20;
tr.load_batch_for_attribute(0,20)
tr.defineWeight(batsize)
tr.define_loss_attribute(fn)


sess=tf.Session()
sess.run(tf.global_variables_initializer())
fn.restore_model(sess,'C:/Users/libar/Desktop/cat_full/2 epoch/final/model') 
conv_4=sess.run(fn.conv_4_3,feed_dict={fn.imgs:tr.batch})
tr.defineWeight(batsize)
fn.get_roi(tr.landmark_x,tr.landmark_y,conv_4,batsize)
feature = sess.run(fn.fc_2,feed_dict={fn.imgs:tr.batch,fn.landmark_visibility:tr.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],tr.output_category:tr.cat_prob,fn.keep_prob:0.5})
triplet_loss=sess.run(tr.loss_triplet,feed_dict={fn.imgs:tr.batch,fn.landmark_visibility:tr.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],tr.output_category:tr.cat_prob,tr.pos:tr.pos_feed,tr.neg:tr.neg_feed,fn.keep_prob:0.5})
print(triplet_loss)    
"""