# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import FashionNet_VGG as fashionnet
import random
import os

class trainer:
    def __init__(self,model_type,batSize=20):
        self.RGB_MEAN = np.array([[ 103.939, 116.779, 123.68 ]],dtype=np.float32)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.param=[]
        self.out=[]
        self.batSize=batSize
        self.model_type = model_type        
        self.img_list=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        self.img_stack=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        if model_type is 'full':
            self.num_of_cat = 6
            #self.img_cat=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,1,2,-1,3,-1,-1,4,5,-1,-1]
            self.img_cat=[38,40,41,43,46,47]
        elif model_type is 'upper':
            self.num_of_cat = 17
            #self.img_cat=[0,1,2,3,4,5,6,-1,7,8,9,10,11,-1,12,13,14,15,16,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.img_cat=[0,1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]
        else:
            self.num_of_cat = 12
            #self.img_cat=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,-1,3,4,-1,5,6,-1,7,8,9,10,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.img_cat=[21,22,23,25,26,28,29,31,32,33,34,35]
            
            

    """
    image 로딩, 리사이징
    output 로딩
    
    img=test.load_image('C:/Users/libar/Desktop/Landmark Prediction/img/img_00000001.jpg')
    """

    def load_batch(self):
        
        imgDir=('C:/Users/libar/Desktop/Attribute Prediction/')
        self.batch = np.zeros((self.batSize,224,224,3),dtype=np.float32) # batch 초기화
        self.cat_prob = np.zeros((self.batSize,self.num_of_cat),dtype=np.float32)
        self.cat = []       
        imgList =[]

        
        for i in range(self.batSize):
            cat_out= random.randrange(0,self.num_of_cat)
            cat = self.img_cat[cat_out]
            
            if len(self.img_stack[cat]) is 0:
                for j in range(len(self.img_list[cat])):
                    self.img_stack[cat].append(self.img_list[cat][j])
                random.shuffle(self.img_stack[cat])
            
            top =self.img_stack[cat].pop()
            imgList.append(self.annoList[top][1])
            self.cat.append(cat_out)
            self.cat_prob[i][cat_out]=1
       
        for i in range(self.batSize):
            imageFileName = imgDir+imgList[i]
            img = Image.open(imageFileName)
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
    
    def readCsv(self):
        self.annoList=[]
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/img_cat_val.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.annoList.append(i)
        f.close()
        
        self.arr=np.array(self.annoList)
        
        for i in range(self.arr.shape[0]):
            if int(self.arr[i][3]) is 0:
                self.img_list[int(self.arr[i][2])-1].append(int(self.arr[i][0]))
                self.img_stack[int(self.arr[i][2])-1].append(int(self.arr[i][0]))\
                
        for i in range(50):
            random.shuffle(self.img_stack[i])

    
   
    def define_loss(self,fn):
        self.output_category=tf.placeholder(tf.float32, [self.batSize, self.num_of_cat])
        self.loss = tf.losses.softmax_cross_entropy(self.output_category,fn.cat_prob)
    """
    def train(self):
        self.readCsv()

        fn=fashionnet.FashionNet()
        fn.build_net(model_type='full',Dropout=True)
        self.define_loss(fn)

        #learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 5 epoch)
        learningRate = 0.00001
        train = tf.train.AdamOptimizer(learningRate).minimize(self.loss)
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        fn.restore_model(sess,'C:/Users/libar/Desktop/cat_full/init/model') 
        print('--------------------------------------------------------------')                
                
        for i in range(50000):
            self.load_batch()
            sess.run(train,feed_dict={fn.img:self.batch,self.output_category:self.cat_prob,fn.keep_prob:0.5})
            #if i%50 is 0:
            [l1,out]=sess.run([self.loss,fn.cat_prob],feed_dict={fn.img:self.batch,self.output_category:self.cat_prob,fn.keep_prob:0.5})
            print('< ',str(i),' batch, ','번째 batch >')
            print('loss: ',l1)
            print(np.array(self.cat))
            print(np.argmax(out,1))
            print(out)
            print('--------------------------------------------------------------')
            if i%1000 is 0:
                dirname = 'C:/Users/libar/Desktop/cat_full/'+str(i)+' batch'
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(i)+' batch/model')
            
        dirname ='C:/Users/libar/Desktop/cat_full/final/model'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)            
        fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/final/model')
        """
        
    def train(self):
        self.readCsv()

        fn=fashionnet.FashionNet()
        fn.build_net(model_type='lower',Dropout=True)
        self.define_loss(fn)

        #learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 5 epoch)
        #full: 0.00001 10000 batch
        #upper: 0.000005 30000batch
        learningRate = 0.000005
        train = tf.train.AdamOptimizer(learningRate).minimize(self.loss)
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        fn.restore_model(sess,'C:/Users/libar/Desktop/cat_lower/init/model') 
        print('--------------------------------------------------------------')                
                
        for i in range(70000):
            self.load_batch()
            sess.run(train,feed_dict={fn.img:self.batch,self.output_category:self.cat_prob,fn.keep_prob:0.5})
            #if i%50 is 0:
            [l1,out]=sess.run([self.loss,fn.cat_prob],feed_dict={fn.img:self.batch,self.output_category:self.cat_prob,fn.keep_prob:0.5})
            print('< ',str(i),' batch, ','번째 batch >')
            print('loss: ',l1)
            print(np.array(self.cat))
            print(np.argmax(out,1))
            print(out)
            print('--------------------------------------------------------------')
            if i%1000 is 0:
                dirname = 'C:/Users/libar/Desktop/cat_lower/'+str(i)+' batch'
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                fn.save_model(sess,'C:/Users/libar/Desktop/cat_lower/'+str(i)+' batch/model')
            
        dirname ='C:/Users/libar/Desktop/cat_lower/final/model'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)            
        fn.save_model(sess,'C:/Users/libar/Desktop/cat_lower/final/model')


tr=trainer('lower')
tr.train()