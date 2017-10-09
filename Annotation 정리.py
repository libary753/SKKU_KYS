# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 05:23:52 2017

@author: libar
"""
import csv
        
def fix_attribute():
    csvList = []
    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Attr.csv','r')
    csvReader = csv.reader(f)
    for i in csvReader:  
        newLine=i[0]
        while newLine.find('  ')!=-1:
            newLine=newLine.replace('  ',' ')
        csvList+=[newLine]
    f.close()

    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Attr_only.csv','w', encoding='utf-8', newline='')
    csvWriter=csv.writer(f,delimiter=',')
    for i in range(len(csvList)):
        newLine=csvList[i]
        newLine=newLine[newLine.find(' ')+1:]
        csvWriter.writerow([newLine])
    f.close()

def fix_category():
    csvList = []
    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Cat.csv','r')
    csvReader = csv.reader(f)
    for i in csvReader:  
        newLine=i[0]
        while newLine.find('  ')!=-1:
            newLine=newLine.replace('  ',' ')
        csvList+=[newLine]
    f.close()

    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Cat_only.csv','w', encoding='utf-8', newline='')
    csvWriter=csv.writer(f,delimiter=',')
    for i in range(len(csvList)):
        newLine=csvList[i]
        newLine=newLine[newLine.find(' ')+1:]
        csvWriter.writerow([newLine])
    f.close()
    
def fix_landmark():
    csvList = []
    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Land.csv','r')
    csvReader = csv.reader(f)
    for i in csvReader:  
        newLine=i[0]
        while newLine.find('  ')!=-1:
            newLine=newLine.replace('  ',' ')
        csvList+=[newLine]
    f.close()

    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Land_only.csv','w', encoding='utf-8', newline='')
    csvWriter=csv.writer(f,delimiter=',')
    for i in range(len(csvList)):
        csvWriter.writerow([csvList[i]])
    f.close()
    
def fix_img():
    csvList = []
    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Img.csv','r')
    csvReader = csv.reader(f)
    for i in csvReader:
        csvList+=[i[0]]
    f.close()

    f = open('C:/Users/libar/Desktop/Attribute Prediction/Anno/Img_split.csv','w', encoding='utf-8', newline='')
    csvWriter=csv.writer(f,delimiter=',')
    for i in range(len(csvList)):
        spl=csvList[i].split('/')
        csvWriter.writerow([csvList[i],spl[1],spl[2]])
    f.close()