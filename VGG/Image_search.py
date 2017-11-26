# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import FashionNet_retrieval as fashionnet_retrieval
import FashionNet_New_Cat as fashionnet_category
import FashionNet_Landmark as fashionnet_landmark

class test:
    def __init__(self,model_type):
        self.batch = np.zeros((1,224,224,3),dtype=np.float32)
        self.RGB_MEAN = np.array([[ 103.939, 116.779, 123.68 ]],dtype=np.float32)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.model_type = model_type        

        if model_type is 'full':
            self.img_cat=[38,40,41,43,46,47]
        elif model_type is 'upper':
            self.img_cat=[0,1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]
        else:
            self.img_cat=[21,22,23,25,26,28,29,31,32,33,34,35]
            
        
    """
    이미지 불러오기
    
    """

    def load_image(self,path):
    
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
    

img_path=[]

f = open('C:/Users/libar/Desktop/img_path.csv','r')
csvReader = csv.reader(f)
for i in csvReader:          
    img_path.append(i)
f.close()

feat = np.zeros([len(img_path),4096],dtype=np.float32)

print('Read Feature File')
f = open('C:/Users/libar/Desktop/features.csv','r')
csvReader = csv.reader(f)
raw=0
for i in csvReader:
    if raw%1000 is 0:
        print(raw)
    for j in range(4096):
        feat[raw][j]=float(i[j])
    raw = raw+1
f.close()

img_path=[]

f = open('C:/Users/libar/Desktop/img_path.csv','r')
csvReader = csv.reader(f)
for i in csvReader:          
    img_path.append(i)
f.close()
   
path = np.array(img_path)
 
model_type='upper'
#i_path='C:/Users/libar/Desktop/image_test/yellow_tee.jpg'
imgDir = ('C:/Users/libar/Desktop/Attribute Prediction/')
i_path=imgDir+img_path[1][0]
image = Image.open(i_path)
#i_idx = 207
#i_path='C:/Users/libar/Desktop/Attribute Prediction/'+img_path[i_idx][0]
ts=test(model_type)
ts.load_image(i_path)



tf.reset_default_graph()
sess=tf.Session()
fn_lnd=fashionnet_landmark.FashionNet(model_type)
fn_lnd.build_net()
fn_lnd.restore_model(sess,'C:/Users/libar/Desktop/weight/land/'+model_type+'/model')

[l_x,l_y,l_v_prob]=sess.run([fn_lnd.out_landmark_x,fn_lnd.out_landmark_y,fn_lnd.visibility],feed_dict={fn_lnd.img:ts.batch})
l_v=[np.argmax(l_v_prob,1)]

tf.reset_default_graph()
sess=tf.Session()

fn_cat=fashionnet_category.FashionNet(model_type)
fn_cat.build_net()
fn_cat.restore_model(sess,'C:/Users/libar/Desktop/weight/cat/'+model_type+'/model')

conv_4=sess.run(fn_cat.conv_4_3,feed_dict={fn_cat.img:ts.batch})
roi=fn_cat.get_roi(l_x,l_y,l_v,conv_4,1,model_type,sess)

cat = sess.run([fn_cat.cat_prob],feed_dict={fn_cat.img:ts.batch,fn_cat.landmark_visibility:l_v,fn_cat.pool_landmark:roi})
cat_idx=ts.img_cat[np.argmax(cat)]
tf.reset_default_graph()
sess=tf.Session()

fn_ret=fashionnet_retrieval.FashionNet(model_type)
fn_ret.build_net()
fn_ret.restore_model(sess,'C:/Users/libar/Desktop/weight/ret/'+model_type+'/model')

conv_4=sess.run(fn_ret.conv_4_3,feed_dict={fn_ret.img:ts.batch})
fn_ret.get_roi(l_x,l_y,conv_4,1,ts.model_type)

if model_type is 'full':
    fc_7 = sess.run([fn_ret.fc_2],feed_dict={fn_ret.img:ts.batch,fn_ret.landmark_visibility:l_v,fn_ret.landmark_1:fn_ret.landmark_roi[:,0],fn_ret.landmark_2:fn_ret.landmark_roi[:,1],fn_ret.landmark_3:fn_ret.landmark_roi[:,2],fn_ret.landmark_4:fn_ret.landmark_roi[:,3],fn_ret.landmark_5:fn_ret.landmark_roi[:,4],fn_ret.landmark_6:fn_ret.landmark_roi[:,5],fn_ret.landmark_7:fn_ret.landmark_roi[:,6],fn_ret.landmark_8:fn_ret.landmark_roi[:,7]}) 
elif model_type is 'upper':
    fc_7 = sess.run([fn_ret.fc_2],feed_dict={fn_ret.img:ts.batch,fn_ret.landmark_visibility:l_v,fn_ret.landmark_1:fn_ret.landmark_roi[:,0],fn_ret.landmark_2:fn_ret.landmark_roi[:,1],fn_ret.landmark_3:fn_ret.landmark_roi[:,2],fn_ret.landmark_4:fn_ret.landmark_roi[:,3],fn_ret.landmark_5:fn_ret.landmark_roi[:,4],fn_ret.landmark_6:fn_ret.landmark_roi[:,5]}) 
elif model_type is 'lower':
    fc_7 = sess.run([fn_ret.fc_2],feed_dict={fn_ret.img:ts.batch,fn_ret.landmark_visibility:l_v,fn_ret.landmark_1:fn_ret.landmark_roi[:,0],fn_ret.landmark_2:fn_ret.landmark_roi[:,1],fn_ret.landmark_3:fn_ret.landmark_roi[:,2],fn_ret.landmark_4:fn_ret.landmark_roi[:,3]}) 

fc_7 = np.array(fc_7[0][0])

dist = []

for i in range(len(feat)):
    distance = np.sum(np.square(feat[i]-fc_7))
    dist.append(distance)

idx=[]

for i in range(15):
    idx.append(np.argmin(dist))
    dist[idx[i]]= np.finfo(np.float32).max

imgDir = ('C:/Users/libar/Desktop/Attribute Prediction/')
result = []
for i in range(15):
    result.append(imgDir + path[idx[i]][0])

img_list = []
count = 0
for i in result:
    img_list.append(Image.open(i))
    Image.open(i).show()