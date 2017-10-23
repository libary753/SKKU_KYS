# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 01:32:53 2017

@author: libar
"""

import numpy as np
import csv

f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Land_lower.csv','r')

landList=[]

csvReader = csv.reader(f)
for i in csvReader:          
    landList.append(i)
    
f.close()

arr=np.array(landList)

attrList = []

f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/Attr.csv','r')
csvReader = csv.reader(f)
for i in csvReader:
    attrList+=[i[0]]
f.close()

arr = np.zeros(dtype=np.int32,shape=[1000,3])

for i in landList:
    index = int(i[0])
    attr = list(map(int,attrList[index].split(' ')))
    for j in range(1000):
        arr[j][attr[j]+1]=arr[j][attr[j]+1]+1
        
np.savetxt('C:/Users/libar/Desktop/Attribute Prediction/Anno/Final/count_lower.csv',arr,delimiter=',')
