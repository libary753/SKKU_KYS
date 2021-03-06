# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import FashionNet_Full_1st_stage as fashionnet_1
import FashionNet_Attribute_Prediction as fashionnet_2
import FashionNet_Attribute_Prediction_for_training as fashionnet_3

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
    def load_batch_for_landmark(self,batNum,model_type,batSize=50):
        self.batSize=batSize
        self.batch = np.zeros((batSize,224,224,3),dtype=np.float32) # batch 초기화
    
        imgDir=('C:/Users/libar/Desktop/Attribute Prediction/')
        self.setOutput_landmark(batNum,batSize,model_type)
        
        for i in range(batSize):
            imageFileName = imgDir+self.img_list[batNum*batSize+i]
            img = Image.open(imageFileName)
            self.output[2][i]=self.output[2][i]/img.size[0]-0.5
            self.output[3][i]=self.output[3][i]/img.size[1]-0.5
            img = self.norm_image(img)
            self.batch[i]=img
    
    def load_batch_for_attribute(self,batNum,batSize=50):
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
    
    def readCsv_attribute(self,model_type):
        csvList = []
        f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Anno_'+model_type+'.csv','r')
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
        self.category_list=arr[:,2]
        if model_type is 'full':
            self.visibility_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'),arr[:,22].astype('float32'),arr[:,25].astype('float32'))).T
            self.x_list=np.vstack((arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'),arr[:,23].astype('float32'),arr[:,26].astype('float32'))).T
            self.y_list=np.vstack((arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'),arr[:,24].astype('float32'),arr[:,27].astype('float32'))).T
    
        elif model_type is 'upper':
            self.visibility_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'),arr[:,16].astype('float32'),arr[:,19].astype('float32'))).T
            self.x_list=np.vstack((arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'),arr[:,17].astype('float32'),arr[:,20].astype('float32'))).T
            self.y_list=np.vstack((arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'),arr[:,18].astype('float32'),arr[:,21].astype('float32'))).T
    
        else:
            self.visibility_list=np.vstack((arr[:,4].astype('float32'),arr[:,7].astype('float32'),arr[:,10].astype('float32'),arr[:,13].astype('float32'))).T
            self.x_list=np.vstack((arr[:,5].astype('float32'),arr[:,8].astype('float32'),arr[:,11].astype('float32'),arr[:,14].astype('float32'))).T
            self.y_list=np.vstack((arr[:,6].astype('float32'),arr[:,9].astype('float32'),arr[:,12].astype('float32'),arr[:,15].astype('float32'))).T
    
    def readCsv_landmark(self,model):
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
        
    def setOutput_landmark(self,batNum,batSize,model_type):
        
        if model_type is 'full':
            num_of_points=8
        elif model_type is 'upper':
            num_of_points=6
        else:
            num_of_points=4
            
        visibility_output=self.visibility_list[batSize*batNum:batSize*(batNum+1),:] 
        visibility_prob=np.zeros((batSize,num_of_points,3))
        for i in range(batSize):
            for j in range(num_of_points):
                visibility_prob[i][j][int(self.visibility_list[batSize*batNum+i][j])]=1
        x_output=self.x_list[batSize*batNum:batSize*(batNum+1),:] 
        y_output=self.y_list[batSize*batNum:batSize*(batNum+1),:] 
        self.output=[visibility_output,visibility_prob,x_output,y_output,]
        self.img_name=self.img_list[batSize*batNum:batSize*(batNum+1)]
        
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
    
    def define_loss_landmark(self,fn,batSize,model_type):
        
        if model_type is 'full':
            num_of_points=8
        elif model_type is 'upper':
            num_of_points=6
        else:
            num_of_points=4
            
        self.output_vis = tf.placeholder(tf.float32, [batSize, num_of_points])
        self.output_vis_prob = tf.placeholder(tf.float32, [batSize,num_of_points,3])
        self.output_x = tf.placeholder(tf.float32, [batSize, num_of_points])
        self.output_y = tf.placeholder(tf.float32, [batSize, num_of_points])

        self.loss_landmark=tf.reduce_mean(tf.matmul(tf.sqrt(tf.square(self.output_x - fn.out_landmark_x)+tf.square(self.output_y - fn.out_landmark_y)),tf.transpose(self.output_vis)))

        if model_type is 'full':
            self.loss_visibility_1=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,0,:], fn.out_visibility_1)
            self.loss_visibility_2=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,1,:], fn.out_visibility_2)
            self.loss_visibility_3=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,2,:], fn.out_visibility_3)
            self.loss_visibility_4=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,3,:], fn.out_visibility_4)
            self.loss_visibility_5=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,4,:], fn.out_visibility_5)
            self.loss_visibility_6=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,5,:], fn.out_visibility_6)
            self.loss_visibility_7=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,6,:], fn.out_visibility_7)
            self.loss_visibility_8=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,7,:], fn.out_visibility_8)
            self.loss_visibility = self.loss_visibility_1+self.loss_visibility_2+self.loss_visibility_3+self.loss_visibility_4+self.loss_visibility_5+self.loss_visibility_6+self.loss_visibility_7+self.loss_visibility_8
        elif model_type is 'upper':
            self.loss_visibility_1=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,0,:], fn.out_visibility_1)
            self.loss_visibility_2=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,1,:], fn.out_visibility_2)
            self.loss_visibility_3=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,2,:], fn.out_visibility_3)
            self.loss_visibility_4=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,3,:], fn.out_visibility_4)
            self.loss_visibility_5=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,4,:], fn.out_visibility_5)
            self.loss_visibility_6=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,5,:], fn.out_visibility_6)
            self.loss_visibility = self.loss_visibility_1+self.loss_visibility_2+self.loss_visibility_3+self.loss_visibility_4+self.loss_visibility_5+self.loss_visibility_6
        else:
            self.loss_visibility_1=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,0,:], fn.out_visibility_1)
            self.loss_visibility_2=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,1,:], fn.out_visibility_2)
            self.loss_visibility_3=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,2,:], fn.out_visibility_3)
            self.loss_visibility_4=tf.losses.softmax_cross_entropy(self.output_vis_prob[:,3,:], fn.out_visibility_4)
            self.loss_visibility = self.loss_visibility_1+self.loss_visibility_2+self.loss_visibility_3+self.loss_visibility_4
        
    
    def define_loss_attribute(self,fn,batSize):
        self.output_category = tf.placeholder(tf.float32, [batSize, 50])
        self.loss_category = tf.losses.softmax_cross_entropy(self.output_category,fn.out_category_prob)
        
    def train_landmark(self):
        test.readCsv_attribute('upper')
        fn=fashionnet_1.FashionNet_1st()
        fn.build_net('upper',Dropout=True)
        batsize=20
        self.define_loss_landmark(fn,batsize,'upper')
        #loss weight: 처음에는 1:10, 나중에는 1:20
        #loss = self.loss_landmark+10*self.loss_visibility
        #learningRate= 0.0001(5 epoch까지) 0.00001(그 다음 5 epoch)
        learningRate = 0.00001
        #learningRate = 0.00001
        #learningRate = 0.000001
        #10 epoch 이후에 5:1
        loss = self.loss_landmark + self.loss_visibility
        #loss = 5*self.loss_landmark + self.loss_visibility
        
        train = tf.train.AdamOptimizer(learningRate).minimize(loss)
        
        
        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        fn.restore_model(sess,'C:/Users/libar/Desktop/save_upper/5 epoch/final/model') 
        print('--------------------------------------------------------------')                
                
        for j in range(5,11):
            self.readCsv_attribute('upper')
            for i in range(5049):
                self.load_batch_for_landmark(i,'upper',batsize)
                
                sess.run([train],feed_dict={fn.img:self.batch,fn.keep_prob:0.5,self.output_vis:self.output[0]%2,self.output_vis_prob:self.output[1],self.output_x:self.output[2],self.output_y:self.output[3]})
                if i%50 is 0:
                    [l1,l2,out]=sess.run([self.loss_landmark,self.loss_visibility,fn.out],feed_dict={fn.img:self.batch,fn.keep_prob:0.5,self.output_vis:self.output[0]%2,self.output_vis_prob:self.output[1],self.output_x:self.output[2],self.output_y:self.output[3]})
                    print('< ',str(j),' epoch, ',str(i),'번째 batch >')
                    print('landmark loss: ',l1)
                    print('visibility loss: ',l2)
                    print(out[0][0])
                    print('..............................................................')
                    print(test.output[2][0])
                    print('--------------------------------------------------------------')
                    if i%400 is 0:
                        fn.save_model(sess,'C:/Users/libar/Desktop/save_upper/'+str(j)+' epoch/'+str(i)+'/model')
                        
            fn.save_model(sess,'C:/Users/libar/Desktop/save_upper/'+str(j)+' epoch/final/model')
    
    def train_attribute(self):
        test.readCsv_attribute('full')
        fn=fashionnet_3.FashionNet_2nd()
        fn.build_net(Dropout=True)
        batsize=25
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
                conv_4=sess.run(fn.conv_4_3,feed_dict={fn.imgs:test.batch})
                fn.get_roi(test.landmark_x,test.landmark_y,conv_4,batsize)
                sess.run(train,feed_dict={fn.imgs:test.batch,fn.landmark_visibility:test.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.output[1],fn.keep_prob:0.5})
                if i%50 is 0:
                    [l1,out]=sess.run([loss,fn.out_category_prob],feed_dict={fn.imgs:test.batch,fn.landmark_visibility:test.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7],self.output_category:self.output[1],fn.keep_prob:0.5})
                    print('< ',str(j),' epoch, ',str(i),'번째 batch >')
                    print('loss: ',l1)
                    print(test.output[0])
                    print(out[0])
                    print('--------------------------------------------------------------')
                    if i%400 is 0:
                        fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(j)+' epoch/'+str(i)+'/model')
                        
            fn.save_model(sess,'C:/Users/libar/Desktop/cat_full/'+str(j)+' epoch/final/model')
            

test=test()
test.readCsv_attribute('full')
batsize=5
test.load_batch_for_attribute(25,batsize)
fn=fashionnet_3.FashionNet_2nd()
fn.build_net()
sess=tf.Session()
fn.restore_model(sess,'C:/Users/libar/Desktop/cat_full/init/model')
sess.run(tf.global_variables_initializer())
conv_4=sess.run(fn.conv_4_3,feed_dict={fn.imgs:test.batch})
fn.get_roi(test.landmark_x,test.landmark_y,conv_4,batsize)
result=sess.run(fn.out_category_prob,feed_dict={fn.imgs:test.batch,fn.landmark_visibility:test.landmark_v,fn.landmark_1:fn.landmark_roi[:,0],fn.landmark_2:fn.landmark_roi[:,1],fn.landmark_3:fn.landmark_roi[:,2],fn.landmark_4:fn.landmark_roi[:,3],fn.landmark_5:fn.landmark_roi[:,4],fn.landmark_6:fn.landmark_roi[:,5],fn.landmark_7:fn.landmark_roi[:,6],fn.landmark_8:fn.landmark_roi[:,7]})

"""
fn1=fashionnet_1.FashionNet_1st()
fn2=fashionnet_2.FashionNet_2nd()
test=test()
test.readCsv('full')
test.load_batch_for_landmark(0,2)
sess=tf.Session()
fn1.build_net()
fn2.build_net()
sess.run(tf.global_variables_initializer())
fn1.restore_model(sess,'C:/Users/libar/Desktop/final/final/model')
[l,conv4]=sess.run([fn1.conv_4_3],feed_dict={fn1.imgs:test.batch})
"""           