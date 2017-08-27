# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import FashionNet_Full_1st_stage as fashionnet

class test:
    def __init__(self):
        self.RGB_MEAN = np.array([[ 103.939, 116.779, 123.68 ]],dtype=np.float32)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.param=[]
        self.out=[]
        self.batSize=50

    """
    image 로딩, 리사이징
    output 로딩
    
    img=test.load_image('C:/Users/libar/Desktop/Landmark Prediction/img/img_00000001.jpg')
    """
    def load_batch_for_landmark(self,batNum,batSize=50):
        self.batSize=batSize
        self.batch = np.zeros((batSize,224,224,3),dtype=np.float32) # batch 초기화
    
        imgDir=('C:/Users/libar/Desktop/Landmark Prediction/')
        self.setOutput(batNum,batSize)
        
        for i in range(batSize):
            imageFileName = imgDir+self.img_list[batNum*batSize+i]
            img = Image.open(imageFileName)
            self.output[2][i]=self.output[2][i]/img.size[0]-0.5
            self.output[3][i]=self.output[3][i]/img.size[1]-0.5
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
    
    def show_result(self,img_path,prediction_x,prediction_y,model_type):
        img=Image.open(img_path)
        (x,y)=img.size
        result=img
        prediction_x=(prediction_x+0.5)*x
        prediction_y=(prediction_y+0.5)*y
        img_new=Image.new('RGB',(9,9),(247,250,110))
        if model_type is 'full':
            num_of_points=8
        elif model_type is 'upper':
            num_of_points=6
        else:
            num_of_points=4
        for i in range(num_of_points):
            prediction_x[0][i]=max(4,prediction_x[0][i])
            prediction_x[0][i]=min(y-5,prediction_x[0][i])
            prediction_y[0][i]=max(4,prediction_y[0][i])
            prediction_y[0][i]=min(y-5,prediction_y[0][i])
            result.paste(img_new,(int(prediction_x[0][i]),int(prediction_y[0][i])))
        return result
            
    
    """
    csv에서 파일 정보 읽어오기
    """
    def readCsv(self,model):
        csvList = []

        f = open('C:/Users/libar/Desktop/Landmark Prediction/Anno/Anno_'+model+'.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            csvList.append(i)
        f.close()
        
        arr=np.array(csvList)
        
        for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if len(arr[i][j]) is 0:
                        arr[i][j]='0'
        
        self.img_list=arr[:,0]
        self.visibility_list=np.vstack((arr[:,3].astype('float32'),arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'),arr[:,24].astype('float32'))).T
        self.x_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'),arr[:,22].astype('float32'),arr[:,25].astype('float32'))).T
        self.y_list=np.vstack((arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'),arr[:,23].astype('float32'),arr[:,26].astype('float32'))).T
        
    def setOutput(self,batNum,batSize):
        visibility_output=self.visibility_list[batSize*batNum:batSize*(batNum+1),:] 
        visibility_prob=np.zeros((batSize,8,3))
        for i in range(batSize):
            for j in range(8):
                visibility_prob[i][j][int(self.visibility_list[batSize*batNum+i][j])]=1
        x_output=self.x_list[batSize*batNum:batSize*(batNum+1),:] 
        y_output=self.y_list[batSize*batNum:batSize*(batNum+1),:] 
        self.output=[visibility_output,visibility_prob,x_output,y_output,]
        self.img_name=self.img_list[batSize*batNum:batSize*(batNum+1)]
    """
    full model: 50013개 중 44000개: training set 6013개: test set
    """
    
    def define_loss(self,fn,batSize):
        self.output_vis = tf.placeholder(tf.float32, [batSize, 8])
        self.output_vis_prob = tf.placeholder(tf.float32, [batSize,8,3])
        self.output_x = tf.placeholder(tf.float32, [batSize, 8])
        self.output_y = tf.placeholder(tf.float32, [batSize, 8])
        
        self.loss_visibility_1=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,0,:], fn.out_visibility_1)
        self.loss_visibility_2=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,1,:], fn.out_visibility_2)
        self.loss_visibility_3=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,2,:], fn.out_visibility_3)
        self.loss_visibility_4=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,3,:], fn.out_visibility_4)
        self.loss_visibility_5=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,4,:], fn.out_visibility_5)
        self.loss_visibility_6=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,5,:], fn.out_visibility_6)
        self.loss_visibility_7=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,6,:], fn.out_visibility_7)
        self.loss_visibility_8=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,7,:], fn.out_visibility_8)
        
        self.loss_landmark=tf.reduce_mean(tf.matmul(tf.sqrt(tf.square(self.output_x - fn.out_landmark_x)+tf.square(self.output_y - fn.out_landmark_y)),tf.transpose(self.output_vis)))
        self.loss_visibility = self.loss_visibility_1+self.loss_visibility_2+self.loss_visibility_3+self.loss_visibility_4+self.loss_visibility_5+self.loss_visibility_6+self.loss_visibility_7+self.loss_visibility_8
        
    def train(self):
        test.readCsv('full')
        fn=fashionnet.FashionNet()
        fn.build_net(Dropout=True)
        batsize=25
        self.define_loss(fn,batSize=batsize)
        #loss weight: 처음에는 1:10, 나중에는 1:20
        #loss = self.loss_landmark+10*self.loss_visibility
        #learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 3 epoch)
        learningRate = 0.000001
        #3 epoch 이후에 20:1
        loss = 20*self.loss_landmark + self.loss_visibility
        
        train = tf.train.AdamOptimizer(learningRate).minimize(loss)
        
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        fn.restore_model(sess,'C:/Users/libar/Desktop/save_full/10 epoch/final/model')
        print('--------------------------------------------------------------')                
                
        for j in range(7):
            self.readCsv('full')
            for i in range(1776):
                self.load_batch_for_landmark(i,batsize)
                
                sess.run([train],feed_dict={fn.img:self.batch,fn.keep_prob:0.5,self.output_vis:self.output[0]%2,self.output_vis_prob:self.output[1],self.output_x:self.output[2],self.output_y:self.output[3]})
                if i%50 is 0:
                    [l1,l2,out]=sess.run([self.loss_landmark,self.loss_visibility,fn.out],feed_dict={fn.img:self.batch,fn.keep_prob:0.5,self.output_vis:self.output[0]%2,self.output_vis_prob:self.output[1],self.output_x:self.output[2],self.output_y:self.output[3]})
                    print('< ',str(j+11),' epoch, ',str(i),'번째 batch >')
                    print('landmark loss: ',l1)
                    print('visibility loss: ',l2)
                    print(out[0][0])
                    print(out[1][0])
                    print(out[2][0])
                    print('..............................................................')
                    print(test.output[2][0])
                    print(test.output[3][0])
                    print(test.output[1][0][0])
                    print('--------------------------------------------------------------')
                    if i%400 is 0:
                        fn.save_model(sess,'C:/Users/libar/Desktop/save_full/'+str(j+11)+' epoch/'+str(i)+'/model')
                        
            fn.save_model(sess,'C:/Users/libar/Desktop/save_full/'+str(j+11)+' epoch/final/model')