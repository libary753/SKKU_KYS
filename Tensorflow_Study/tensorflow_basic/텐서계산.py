# -*- coding: utf-8 -*-
import tensorflow as tf

a = tf.constant([5,3], name = "input_a") # 텐서로 정의
b = tf.reduce_prod(a,name = "prod_b") # 텐서 성분을 곱해서 스칼라로 반환
c = tf.reduce_sum(a,name = "sum_c") # 텐서 성분을 더해서 스칼라로 반환
d = tf.add(b,c,name = "add_d") # 스칼라합 -> 스칼라계산.py 와 같은 결과

sess = tf.Session()
result=sess.run(d)
writer = tf.summary.FileWriter('./tensor_graph',sess.graph)
sess.close() 