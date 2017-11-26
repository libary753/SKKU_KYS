# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtGui
import tensorflow as tf 
import numpy as np
from PIL import Image
import csv
import FashionNet_retrieval as fashionnet_retrieval
import FashionNet_New_Cat as fashionnet_category
import FashionNet_Landmark as fashionnet_landmark

form_class = uic.loadUiType("FashionNet.ui")[0]



class FashionNet_GUI(QMainWindow, form_class):
    def __init__(self):
        #For GUI
        super().__init__()
        self.setupUi(self)
        self.Button_Open_Image.clicked.connect(self.Button_Open_Image_Clicked)
        self.Button_Load_Weight.clicked.connect(self.Button_Load_Weight_Clicked)
        self.Button_Find_Landmark.clicked.connect(self.Button_Find_Landmark_Clicked)
        self.Button_Find_Category.clicked.connect(self.Button_Find_Category_Clicked)
        self.Button_Search.clicked.connect(self.Button_Search_Clicked)

        #For Fasshion
        self.batch = np.zeros((1,224,224,3),dtype=np.float32)
        self.RGB_MEAN = np.array([[ 103.939, 116.779, 123.68 ]],dtype=np.float32)
        self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
        
        self.model_type = 'upper'        

        if self.model_type is 'full':
            self.img_cat=[38,40,41,43,46,47]
        elif self.model_type is 'upper':
            self.img_cat=[0,1,2,3,4,5,6,8,9,10,11,12,14,15,16,17,18]
        else:
            self.img_cat=[21,22,23,25,26,28,29,31,32,33,34,35]
        
        self.categories = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Flannel', 'Halter', 'Henley', 'Hoodie', 'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck', 'Capris', 'Chinos', 'Culottes', 'Cutoffs', 'Gauchos', 'Jeans', 'Jeggings', 'Jodhpurs', 'Joggers', 'Leggings', 'Sarong', 'Shorts', 'Skirt', 'Sweatpants', 'Sweatshorts', 'Trunks', 'Caftan', 'Cape', 'Coat', 'Coverup', 'Dress', 'Jumpsuit', 'Kaftan', 'Kimono', 'Nightdress', 'Onesie', 'Robe', 'Romper', 'Shirtdress', 'Sundress']

    def Button_Open_Image_Clicked(self):
        path = QFileDialog.getOpenFileName(self)
        self.image_path= path[0]
        self.image = Image.open(self.image_path)
        self.SetImage(self.image,self.Image_Origin)
        self.load_image(self.image_path)
        
        """
        pixmap = QtGui.QPixmap(path[0])
        self.Image_Origin.setPixmap(pixmap)
        """
    
    """    
    def Button_Load_Weight_Clicked(self):
        
        self.img_path=[]

        f = open('C:/Users/libar/Desktop/img_path.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.img_path.append(i)
        f.close()
        
        self.feat = np.zeros([len(self.img_path),4096],dtype=np.float32)
        
        print('Read Feature File')
        f = open('C:/Users/libar/Desktop/features.csv','r')
        csvReader = csv.reader(f)
        raw=0
        for i in csvReader:
            if raw%1000 is 0:
                print(raw)
            for j in range(4096):
                self.feat[raw][j]=float(i[j])
            raw = raw+1
        f.close()
        
        self.img_path=[]
        
        f = open('C:/Users/libar/Desktop/img_path.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            self.img_path.append(i)
        f.close()
           
        self.path = np.array(self.img_path)
        """

    #짧게만들어놓음    
    def Button_Load_Weight_Clicked(self):
        
        self.img_path=[]

        f = open('C:/Users/libar/Desktop/img_path.csv','r')
        csvReader = csv.reader(f)
        for i in csvReader:          
            if i is 1000:
                break;
            self.img_path.append(i)
        f.close()
        
        self.feat = np.zeros([1000,4096],dtype=np.float32)
        
        print('Read Feature File')
        f = open('C:/Users/libar/Desktop/features.csv','r')
        csvReader = csv.reader(f)
        self.raw=0
        for i in csvReader:
            if self.raw - 1000 is 0:
                break;
            for j in range(4096):
                self.feat[self.raw][j]=float(i[j])
            self.raw = self.raw+1
        f.close()
           
        self.path = np.array(self.img_path)
        print('Weight Loading is finished.')

    def Button_Find_Landmark_Clicked(self):
        print("Start landmark detection")
        tf.reset_default_graph()
        sess=tf.Session()
        fn_lnd=fashionnet_landmark.FashionNet(self.model_type)
        fn_lnd.build_net()
        fn_lnd.restore_model(sess,'C:/Users/libar/Desktop/weight/land/'+self.model_type+'/model')
        
        [self.l_x,self.l_y,self.l_v_prob]=sess.run([fn_lnd.out_landmark_x,fn_lnd.out_landmark_y,fn_lnd.visibility],feed_dict={fn_lnd.img:self.batch})
        print(self.l_x)
        print(self.l_y)
        self.l_v=[np.argmax(self.l_v_prob,1)]
        self.show_landmark_result(self.image_path,self.l_x,self.l_y,'upper')

    def Button_Find_Category_Clicked(self):
        print("Start category prediction")
        tf.reset_default_graph()
        sess=tf.Session()
        
        fn_cat=fashionnet_category.FashionNet(self.model_type)
        fn_cat.build_net()
        fn_cat.restore_model(sess,'C:/Users/libar/Desktop/weight/cat/'+self.model_type+'/model')
        
        conv_4=sess.run(fn_cat.conv_4_3,feed_dict={fn_cat.img:self.batch})
        roi=fn_cat.get_roi(self.l_x,self.l_y,self.l_v,conv_4,1,self.model_type,sess)
        
        self.cat = sess.run([fn_cat.cat_prob],feed_dict={fn_cat.img:self.batch,fn_cat.landmark_visibility:self.l_v,fn_cat.pool_landmark:roi})
        self.first_idx=np.argmax(self.cat)
        self.first_catname=self.categories[self.img_cat[self.first_idx]]
        self.cat[0][0][self.first_idx]=0
        self.second_idx=np.argmax(self.cat)
        self.second_catname=self.categories[self.img_cat[self.second_idx]]
        self.cat[0][0][self.second_idx]=0
        self.third_idx=np.argmax(self.cat)
        self.third_catname=self.categories[self.img_cat[self.third_idx]]
        
        self.Category_1.setText('1. ' + self.first_catname)
        self.Category_2.setText('2. ' + self.second_catname)
        self.Category_3.setText('3. ' + self.third_catname)
        
        #self.third_idx=self.img_cat[np.argmax(self.cat)]
        print([self.first_catname,self.second_catname,self.third_catname])

    def Button_Search_Clicked(self):
        tf.reset_default_graph()
        sess=tf.Session()
        
        fn_ret=fashionnet_retrieval.FashionNet(self.model_type)
        fn_ret.build_net()
        fn_ret.restore_model(sess,'C:/Users/libar/Desktop/weight/ret/'+self.model_type+'/model')
        
        conv_4=sess.run(fn_ret.conv_4_3,feed_dict={fn_ret.img:self.batch})
        fn_ret.get_roi(self.l_x,self.l_y,conv_4,1,self.model_type)
        
        if self.model_type is 'full':
            fc_7 = sess.run([fn_ret.fc_2],feed_dict={fn_ret.img:self.batch,fn_ret.landmark_visibility:self.l_v,fn_ret.landmark_1:fn_ret.landmark_roi[:,0],fn_ret.landmark_2:fn_ret.landmark_roi[:,1],fn_ret.landmark_3:fn_ret.landmark_roi[:,2],fn_ret.landmark_4:fn_ret.landmark_roi[:,3],fn_ret.landmark_5:fn_ret.landmark_roi[:,4],fn_ret.landmark_6:fn_ret.landmark_roi[:,5],fn_ret.landmark_7:fn_ret.landmark_roi[:,6],fn_ret.landmark_8:fn_ret.landmark_roi[:,7]}) 
        elif self.model_type is 'upper':
            fc_7 = sess.run([fn_ret.fc_2],feed_dict={fn_ret.img:self.batch,fn_ret.landmark_visibility:self.l_v,fn_ret.landmark_1:fn_ret.landmark_roi[:,0],fn_ret.landmark_2:fn_ret.landmark_roi[:,1],fn_ret.landmark_3:fn_ret.landmark_roi[:,2],fn_ret.landmark_4:fn_ret.landmark_roi[:,3],fn_ret.landmark_5:fn_ret.landmark_roi[:,4],fn_ret.landmark_6:fn_ret.landmark_roi[:,5]}) 
        elif self.model_type is 'lower':
            fc_7 = sess.run([fn_ret.fc_2],feed_dict={fn_ret.img:self.batch,fn_ret.landmark_visibility:self.l_v,fn_ret.landmark_1:fn_ret.landmark_roi[:,0],fn_ret.landmark_2:fn_ret.landmark_roi[:,1],fn_ret.landmark_3:fn_ret.landmark_roi[:,2],fn_ret.landmark_4:fn_ret.landmark_roi[:,3]}) 
        
        fc_7 = np.array(fc_7[0][0])
        
        dist = []
        
        for i in range(len(self.feat)):
            distance = np.sum(np.square(self.feat[i]-fc_7))
            dist.append(distance)
        
        idx=[]
        
        for i in range(15):
            idx.append(np.argmin(dist))
            dist[idx[i]]= np.finfo(np.float32).max
        
        imgDir = ('C:/Users/libar/Desktop/Attribute Prediction/')
        result = []
        for i in range(15):
            result.append(imgDir + self.path[idx[i]][0])
        
        img_list = []
        for i in result:
            img_list.append(Image.open(i))
            
        self.SetImage(img_list[0],self.Image_Search_1)
        self.SetImage(img_list[1],self.Image_Search_2)
        self.SetImage(img_list[2],self.Image_Search_3)
        self.SetImage(img_list[3],self.Image_Search_4)
        self.SetImage(img_list[4],self.Image_Search_5)
        self.SetImage(img_list[5],self.Image_Search_6)
        self.SetImage(img_list[6],self.Image_Search_7)
        self.SetImage(img_list[7],self.Image_Search_8)
        self.SetImage(img_list[8],self.Image_Search_9)
        self.SetImage(img_list[9],self.Image_Search_10)
        
    def SetImage(self, img, panel):
        height = panel.size().height();
        width = panel.size().width();
        
        scale = height/max(img.size)
        s1=round(img.size[0]*scale)
        s2=round(img.size[1]*scale)
        

        img = img.resize((s1,s2))
        img_resized = Image.new('RGB',(height,width))
        img_resized.paste( (255,255,255), [0,0,img_resized.size[0],img_resized.size[1]])
        
        if img.size[0] is height:
            offset=round((height-img.size[1])/2)
            img_resized.paste(img,(0,offset))      
        else:
            offset=round((width-img.size[0])/2)
            img_resized.paste(img,(offset,0))      
            #img_resized.show()
        
        arr = np.array(img_resized)
        pix = QtGui.QPixmap(QtGui.QImage(arr, arr.shape[1],arr.shape[0], arr.shape[1] * 3,QtGui.QImage.Format_RGB888))
        panel.setPixmap(pix)
        
        
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

    def show_landmark_result(self,path,prediction_x,prediction_y,model_type):
        img=Image.open(path)
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
        
        self.SetImage(result,self.Image_Landmark)
            


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = FashionNet_GUI()
    myWindow.show()
    app.exec_()