#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np # 파일로부터 데이터를 불러오는데 사용

"""
numpy loadext 사용
xy = np.loadtxt('TF_Lab_04_02_data.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1] #slicing
y_data = xy[:,[-1]]

x = tf.placeholder(tf.float32,shape =[None,3])
y = tf.placeholder(tf.float32,shape =[None,1])

w = tf.Variable(tf.random_normal([3,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name ='bias')

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)

train = optimizer.minimize(cost)

#실행

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train],feed_dict={x:x_data,y:y_data})
    if step%10 ==0:
        print(step,"Cost: ",cost_val,"\nPrediction:\n",hy_val)
"""

#Queue runner 도 사용가능함