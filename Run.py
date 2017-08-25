# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import datetime

class test:
    def __init__(self):
        self.RGB_MEAN = np.array([[ 102.9801, 115.9465, 122.7717]],dtype=np.float32)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.param=[]
        self.out=[]

    """
    image 로딩, 리사이징
    output 로딩
    
    img=test.load_image('D:/fashion-landmarks-master/data/FLD_full/test_00000001.jpg')
    """
    
    def load_image(self,path):
        RGB_MEAN = np.array([[ 102.9801, 115.9465, 122.7717]],dtype=np.float32)
        img = Image.open(path)
        scale = 224/max(img.size)
        s1=round(img.size[0]*scale)
        s2=round(img.size[1]*scale)
        nump=np.zeros((1,224,224,3),dtype=np.float32)
        
        if img.size[0] is 224 and img.size[1] is 224:
            nump[0]=img-RGB_MEAN
        else:
            img = img.resize((s1,s2))
            img_resized = Image.new('RGB',(224,224))
            if img.size[0] is 224:
                offset=round((224-img.size[1])/2)
                img_resized.paste(img,(0,offset))      
                nump[0]=img_resized-RGB_MEAN
            else:
                offset=round((224-img.size[0])/2)
                img_resized.paste(img,(offset,0))        
                nump[0]=img_resized-RGB_MEAN
        nump=np.swapaxes(nump,1,2)
        nump=nump[:,:,:,(2,1,0)]
        return nump
    
    def show_result(self,img,prediction,model_type):
        landmarks=(prediction+0.5)*max(img.size)
        if model_type is 'full':
            return 0
    
    def get_origin_coordinate(self,img_origin,prediction):
        return 0
    
    def run():
        fn=FashionNet()
        fn.build_net(dropout=True)
        sess=tf.Session()
        #prediction=sess.run(fn.out,feed_dict={fn.img:img})
        
        loss_landmark=tf.reduce_mean(tf.matmul(tf.sqrt(tf.square(prediction.output[3] - self.out_fc_pose_landmark_x_3)+tf.square(self.output[4] - self.out_fc_pose_landmark_y_3)),self.output[2].T))
        loss_visibility=0
        learningRate = 0.00001
        loss = loss_landmark+loss_visibility
        train=tf.train.AdamOptimizer(learningRate).minimize(loss)

        sess.run(tf.global_variables_initializer())

        
        for i in range(100001):
            fn.load_batch_for_landmark(i,1)    
            fn.sess.run(train,feed_dict={fn.imgs:fn.batch,fn.keep_prob:0.5})
            if i%2000 is 0:
                print(str(i),'번째, 데이터 백업. 경로: save/'+str(i)+'/model')
                fn.save_model('D:/save/'+str(i)+'/model')
            if i%100 is 0:
                print(str(i),'번째')