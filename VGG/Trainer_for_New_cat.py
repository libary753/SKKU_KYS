# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import FashionNet_New_Cat as fashionnet
import random
import os

class trainer:
    def __init__(self,model_type,batSize=40):
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
            self.cat_img=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,1,2,-1,3,-1,-1,4,5,-1,-1]
            self.img_cat=[38,40,41,43,46,47]
        elif model_type is 'upper':
            self.num_of_cat = 17
            self.cat_img=[0,1,2,3,4,5,6,-1,7,8,9,10,11,-1,12,13,14,15,16,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.img_cat=[0,1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]
        else:
            self.num_of_cat = 12
            self.cat_img=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,-1,3,4,-1,5,6,-1,7,8,9,10,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            self.img_cat=[21,22,23,25,26,28,29,31,32,33,34,35]
            
        self.output_category=tf.placeholder(tf.float32, [self.batSize, self.num_of_cat])
            
            

    """
    image 로딩, 리사이징
    output 로딩
    
    img=test.load_image('C:/Users/libar/Desktop/Landmark Prediction/img/img_00000001.jpg')
    """

    def load_batch(self):
        
        imgDir=('C:/Users/libar/Desktop/Attribute Prediction/')
        self.batch = np.zeros((self.batSize,224,224,3),dtype=np.float32) # batch 초기화
        self.cat_prob = np.zeros((self.batSize,self.num_of_cat),dtype=np.float32)
        self.x = []
        self.y = []
        self.v = []
        self.cat = []    
        imgList =[]

        
        for i in range(self.batSize):
            cat_out= random.randrange(0,self.num_of_cat)
            cat = self.img_cat[cat_out]
            
            if len(self.img_stack[cat]) is 0:
                for j in range(len(self.img_list[cat])):
                    self.img_stack[cat].append(self.img_list[cat][j])
                random.shuffle(self.img_stack[cat])
            
            idx =self.img_stack[cat].pop()
            imgList.append(self.annoList[idx][1])
            self.cat.append(cat_out)
            self.cat_prob[i][cat_out]=1
            
            
            imageFileName = imgDir+imgList[i]
            img = Image.open(imageFileName)
            
            self.x+=[self.x_list[idx]/img.size[0]-0.5]
            self.y+=[self.y_list[idx]/img.size[1]-0.5]
            self.v+=[self.v_list[idx]]
            
            img = self.norm_image(img)
            self.batch[i]=img
            
    def load_batch_for_test(self,batNum):
        
        imgDir=('C:/Users/libar/Desktop/Attribute Prediction/')
        self.batch = np.zeros((self.batSize,224,224,3),dtype=np.float32) # batch 초기화
        self.cat_prob = np.zeros((self.batSize,self.num_of_cat),dtype=np.float32)
        self.x = []
        self.y = []
        self.v = []
        self.cat = []    
        imgList =[]

        
        for i in range(self.batSize):
            
            idx =self.img_list[self.batSize*batNum+i]
            imgList.append(self.img_list[self.batSize*batNum+i])
            cat_out=self.cat_list[self.batSize*batNum+i]
            self.cat.append(cat_out)
            
            imgPath=self.annoList[imgList[i]][1]
            
            imageFileName = imgDir+imgPath
            img = Image.open(imageFileName)
            
            self.x+=[self.x_list[idx]/img.size[0]-0.5]
            self.y+=[self.y_list[idx]/img.size[1]-0.5]
            self.v+=[self.v_list[idx]]
            
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
            
        self.landList=[]
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Landmark.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.landList.append(i)
        f.close()
        
        
        
        arr=np.array(self.landList)
        
        for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if len(arr[i][j]) is 0:
                        arr[i][j]='0'
        
        if self.model_type is 'full':
            self.v_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'),arr[:,23].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'),arr[:,24].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'),arr[:,22].astype('float32'),arr[:,25].astype('float32'))).T
            
        elif self.model_type is 'upper':
            self.v_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'))).T
            
        elif self.model_type is 'lower':
            self.v_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'))).T

    def readCsv_test(self):
        self.annoList=[]
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/test.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.annoList.append(i)
        f.close()
        
        self.arr=np.array(self.annoList)
        self.img_list=[]
        self.cat_list=[]
        
        
        if self.model_type is 'upper':
            self.model_num=1
        elif self.model_type is 'lower':
            self.model_num=2
        elif self.model_type is 'full':
            self.model_num=3
        
        for i in range(self.arr.shape[0]):
            if int(self.arr[i][3]) is 1 and int(self.arr[i][4]) is self.model_num:
                self.img_list.append(int(self.arr[i][0]))#img_list에 index 저장
                self.cat_list.append(int(self.arr[i][2]))
            
        self.landList=[]
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Landmark.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.landList.append(i)
        f.close()
        
        
        
        arr=np.array(self.landList)
        
        for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if len(arr[i][j]) is 0:
                        arr[i][j]='0'
        
        if self.model_type is 'full':
            self.v_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'),arr[:,23].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'),arr[:,24].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'),arr[:,22].astype('float32'),arr[:,25].astype('float32'))).T
            
        elif self.model_type is 'upper':
            self.v_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'))).T
            
        elif self.model_type is 'lower':
            self.v_list=np.vstack((arr[:,2].astype('float32'),arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'))).T
            self.x_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'))).T
            self.y_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'))).T
   
    def define_loss(self,fn):
        
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

        fn=fashionnet.FashionNet(self.model_type)
        fn.build_net(model_type=self.model_type,Dropout=True)
        self.define_loss(fn)

        #learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 5 epoch)
        #full: 0.00001 10000 batch
        #upper: 0.000005 30000batch
        learningRate = 0.000005
        train = tf.train.AdamOptimizer(learningRate).minimize(self.loss)
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        
        
        fn.restore_model(sess,'C:/Users/libar/Desktop/cat_'+self.model_type+'_pre/init/model') 
        print('--------------------------------------------------------------')                
                
        for i in range(70000):
            self.load_batch()
            conv_4=sess.run(fn.conv_4_3,feed_dict={fn.img:self.batch})
            roi=fn.get_roi(self.x,self.y,self.v,conv_4,self.batSize,self.model_type,sess)
            if self.model_type is 'full':
                sess.run(train,feed_dict={fn.img:self.batch,fn.landmark_visibility:self.v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.cat_prob,fn.pool_landmark:roi,fn.keep_prob:0.5}) 
            elif self.model_type is 'upper':
                sess.run(train,feed_dict={fn.img:self.batch,fn.landmark_visibility:self.v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],self.output_category:self.cat_prob,fn.keep_prob:0.5})
            elif self.model_type is 'lower':
                sess.run(train,feed_dict={fn.img:self.batch,fn.landmark_visibility:self.v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],self.output_category:self.cat_prob,fn.keep_prob:0.5})
            #if i%50 is 0:
            if self.model_type is 'full':
                [l1,out]=sess.run([self.loss,fn.cat_prob],feed_dict={fn.img:self.batch,fn.landmark_visibility:self.v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.cat_prob,fn.pool_landmark:roi,fn.keep_prob:0.5})
            elif self.model_type is 'upper':
                [l1,out]=sess.run([self.loss,fn.cat_prob],feed_dict={fn.img:self.batch,fn.landmark_visibility:self.v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],self.output_category:self.cat_prob,fn.keep_prob:0.5})
            elif self.model_type is 'lower':
                [l1,out]=sess.run([self.loss,fn.cat_prob],feed_dict={fn.img:self.batch,fn.landmark_visibility:self.v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],self.output_category:self.cat_prob,fn.keep_prob:0.5})
 
            print('< ',str(i),' batch, ','번째 batch >')
            print('loss: ',l1)
            print(np.array(self.cat))
            print(np.argmax(out,1))
            print(out)
            print('--------------------------------------------------------------')
            if i%1000 is 0:
                dirname = 'C:/Users/libar/Desktop/cat_'+self.model_type+'_pre/'+str(i)+' batch'
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                fn.save_model(sess,'C:/Users/libar/Desktop/cat_'+self.model_type+'_pre/'+str(i)+' batch/model')
            
        dirname ='C:/Users/libar/Desktop/cat_'+self.model_type+'_pre/final/model'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)            
        fn.save_model(sess,'C:/Users/libar/Desktop/cat_'+self.model_type+'/final/model')
     
        
"""
tr=trainer('full')
tr.train()
"""
"""
tr=trainer('lower')
tr.readCsv()

fn=fashionnet.FashionNet(tr.model_type)
fn.build_net(model_type=tr.model_type,Dropout=True)
tr.define_loss(fn)

#learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 5 epoch)
#full: 0.00005 10000 batch 0.00001 10000batch
#upper: 0.00001 30000batch 
learningRate = 0.000001
train = tf.train.AdamOptimizer(learningRate).minimize(tr.loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())


fn.restore_model(sess,'C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/init/model') 
print('--------------------------------------------------------------')                
acc_list=[]        
for i in range(20000):
    tr.load_batch()
    conv_4=sess.run(fn.conv_4_3,feed_dict={fn.img:tr.batch})
    roi=fn.get_roi(tr.x,tr.y,tr.v,conv_4,tr.batSize,tr.model_type,sess)

    #if i%50 is 0:
    if tr.model_type is 'full':
        [l1,out,t]=sess.run([tr.loss,fn.cat_prob,train],feed_dict={fn.img:tr.batch,fn.landmark_visibility:tr.v,tr.output_category:tr.cat_prob,fn.pool_landmark:roi,fn.keep_prob:0.5})
    elif tr.model_type is 'upper':
        [l1,out,t]=sess.run([tr.loss,fn.cat_prob,train],feed_dict={fn.img:tr.batch,fn.landmark_visibility:tr.v,tr.output_category:tr.cat_prob,fn.pool_landmark:roi,fn.keep_prob:0.5})
    elif tr.model_type is 'lower':
        [l1,out,t]=sess.run([tr.loss,fn.cat_prob,train],feed_dict={fn.img:tr.batch,fn.landmark_visibility:tr.v,tr.output_category:tr.cat_prob,fn.pool_landmark:roi,fn.keep_prob:0.5})
 
    print('< ',str(i),' batch, ','번째 batch >')
    print('loss: ',l1)
    groundtruth=np.array(tr.cat)
    output=np.argmax(out,1)
    count=0;
    for j in range(tr.batSize):
        if(groundtruth[j]==output[j]):
            count = count+1
    accuracy = (count/tr.batSize)*100.
    acc_list+=[accuracy]
    print('gt: ',groundtruth)
    print('op: ',output)
    print('accuracy: ',accuracy,'%')
    print('--------------------------------------------------------------')
    

    if i%1000 is 0:
        dirname = 'C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/'+str(i)+' batch'
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fn.save_model(sess,'C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/'+str(i)+' batch/model')
        
        f=open('C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/acc_list_'+str(i)+' batch.csv','w', encoding='utf-8', newline='')
        csvWriter=csv.writer(f,delimiter=',')
        for i in range(len(acc_list)):
            csvWriter.writerow([acc_list[i]])
        f.close()
    
dirname ='C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/final'
if not os.path.isdir(dirname):
    os.mkdir(dirname)            
fn.save_model(sess,'C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/final/model')

f=open('C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/acc_list_final.csv','w', encoding='utf-8', newline='')
csvWriter=csv.writer(f,delimiter=',')
for i in range(len(acc_list)):
    csvWriter.writerow([acc_list[i]])
f.close()
"""

tr=trainer('upper')
tr.readCsv_test()

fn=fashionnet.FashionNet(tr.model_type)
fn.build_net(model_type=tr.model_type,Dropout=False)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

fn.restore_model(sess,'C:/Users/libar/Desktop/cat_'+tr.model_type+'_new/final/model') 

gt=[]
result1=[]
result2=[]
result3=[]
print('--------------------------------------------------------------')                
acc_list=[]        
for i in range(int(len(tr.img_list)/tr.batSize)):
    tr.load_batch_for_test(i)
    conv_4=sess.run(fn.conv_4_3,feed_dict={fn.img:tr.batch})
    roi=fn.get_roi(tr.x,tr.y,tr.v,conv_4,tr.batSize,tr.model_type,sess)

    #if i%50 is 0:
    if tr.model_type is 'full':
        [out]=sess.run([fn.cat_prob],feed_dict={fn.img:tr.batch,fn.landmark_visibility:tr.v,tr.output_category:tr.cat_prob,fn.pool_landmark:roi})
    elif tr.model_type is 'upper':
        [out]=sess.run([fn.cat_prob],feed_dict={fn.img:tr.batch,fn.landmark_visibility:tr.v,tr.output_category:tr.cat_prob,fn.pool_landmark:roi})
    elif tr.model_type is 'lower':
        [out]=sess.run([fn.cat_prob],feed_dict={fn.img:tr.batch,fn.landmark_visibility:tr.v,tr.output_category:tr.cat_prob,fn.pool_landmark:roi})
 
    print('< ',str(i),' batch, ','번째 batch >')
    
    o1=[]        
    o2=[]        
    o3=[]        
    g = tr.cat
    r1=np.argmax(out,1)
    for i in range(tr.batSize):
        out[i][r1[i]]=0
    
    r2=np.argmax(out,1)
    for i in range(tr.batSize):
        out[i][r2[i]]=0
        
    r3=np.argmax(out,1)
        
    
    for i in range(tr.batSize):
        o1=o1+[tr.img_cat[r1[i]]+1]
        o2=o2+[tr.img_cat[r2[i]]+1]
        o3=o3+[tr.img_cat[r3[i]]+1]
            
    gt=gt+g
    result1=result1+o1
    result2=result2+o2
    result3=result3+o3
    
    count=0;

    print('gt: ',g)
    print('o1: ',o1)
    print('o2: ',o2)
    print('o3: ',o3)
    print('--------------------------------------------------------------')

final = np.array([gt,result1,result2,result3])
f=open('C:/Users/libar/Desktop/test_'+tr.model_type+'_new.csv','w', encoding='utf-8', newline='')
csvWriter=csv.writer(f,delimiter=',')
for i in range(len(final.T)):
    csvWriter.writerow(final.T[i])
f.close()