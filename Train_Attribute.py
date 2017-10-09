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
        self.batSize=50

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
            imageFileName = imgDir+self.img_list[batNum*batSize+i]
            img = Image.open(imageFileName)
            self.landmark_x[i]=self.landmark_x[i]/img.size[0]-0.5
            self.landmark_y[i]=self.landmark_y[i]/img.size[1]-0.5
            img = self.norm_image(img)
            self.batch[i]=img
    
    
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
    
    def readCsv_attribute(self,model_type):
        self.landList=[]
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Land_'+model_type+'.csv','r')
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
        
        if model_type is 'full':
            self.visibility_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'),arr[:,23].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'),arr[:,24].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'),arr[:,22].astype('float32'),arr[:,25].astype('float32'))).T
            
        elif model_type is 'upper':
            self.visibility_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'))).T
            
        else:
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
        category_output=self.category_list[batSize*batNum:batSize*(batNum+1)]
        category_prob=np.zeros((batSize,50))
        for i in range(batSize):
            category_prob[i][int(self.category_list[batSize*batNum+i])-1]=1
        
        self.landmark_v=self.visibility_list[batSize*batNum:batSize*(batNum+1),:]%2
        self.landmark_x=self.x_list[batSize*batNum:batSize*(batNum+1),:] 
        self.landmark_y=self.y_list[batSize*batNum:batSize*(batNum+1),:]
        self.output=[category_output,category_prob]
        self.img_name=self.img_list[batSize*batNum:batSize*(batNum+1)]
    
    def define_loss_attribute(self,fn,batSize=20,margin=0.01):
        self.output_category = tf.placeholder(tf.float32, [batSize, 50])
        self.feature = tf.placeholder(tf.float32, [batSize, 4096])
        self.pos = tf.placeholder(tf.int32, [batSize])
        self.neg = tf.placeholder(tf.int32, [batSize])
        self.loss_category = tf.losses.softmax_cross_entropy(self.output_category,fn.out_category_prob)

        feature_pos = tf.gather(self.feature,self.pos)
        feature_neg = tf.gather(self.feature,self.neg)
        
        pos = self.sqr_dist(fn.fc_2, feature_pos)
        neg = self.sqr_dist(fn.fc_2, feature_neg)
        
        self.loss_triplet = pos-neg+margin
        
                    
    def sqr_dist(self,p1,p2): 
        return tf.reduce_sum(tf.square(tf.subtract(p1, p2)))
    
    def train_attribute(self):
        self.readCsv_attribute('full')
        fn=FashionNet.FashionNet_2nd()
        fn.build_net(Dropout=True)
        batsize=20
        self.define_loss_attribute(fn,batSize=batsize)
        #learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 5 epoch)
        learningRate = 0.0001
        loss = self.loss_category
        
        train = tf.train.AdamOptimizer(learningRate).minimize(loss)
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        fn.restore_model(sess,'C:/Users/libar/Desktop/cat_full/init/model') 
        print('--------------------------------------------------------------')                
                
        for j in range(1,6):
            self.readCsv_attribute('full')
            for i in range(2620):
                self.load_batch_for_attribute(i,batsize)
                conv_4=sess.run(fn.conv_4_3,feed_dict={fn.imgs:self.batch})
                fn.get_roi(self.landmark_x,self.landmark_y,conv_4,batsize)
                feature = sess.run(fn.fc_2,feed_dict={fn.imgs:self.batch,fn.landmark_visibility:self.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.output[1],fn.keep_prob:0.5})
                sess.run(train,feed_dict={fn.imgs:self.batch,fn.landmark_visibility:self.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.output[1]fn.keep_prob:0.5})
                if i%50 is 0:
                    [l1,out]=sess.run([loss,fn.out_category_prob],feed_dict={fn.imgs:self.batch,fn.landmark_visibility:self.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.output[1],fn.keep_prob:0.5})
                    print('< ',str(j),' epoch, ',str(i),'번째 batch >')
                    print('loss: ',l1)
                    print(self.output[0])
                    print(out[0])
                    print('--------------------------------------------------------------')
                    if i%400 is 0:
                        fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(j)+' epoch/'+str(i)+'/model')
                        
            fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(j)+' epoch/final/model')
            
            
 
        
fn=FashionNet.FashionNet_2nd()
tr = trainer()
fn.build_net()
tr.define_loss_attribute(fn)