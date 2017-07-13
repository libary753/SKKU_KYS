import tensorflow as tf

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant ("Hello, TensorFlow!")

# seart a  TF sessioni
sess = tf.Session()

#run the op and get result
print(sess.run(hello))


#Build Graph
node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2) # same as node3 = node1 + node2

#Feed data and run graph
sess = tf.Session()

print("sess.run(node1, node2):", sess.run([node1, node2]))
print("sess.run(node3):",sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b

print(sess.run(adder_node,feed_dict={a:3,b:4.5}))
print(sess.run(adder_node,feed_dict={a:[1,3],b:[2,4]}))
print(a)
print(b)
t=[[1,2,3],[4,5,6],[7,8,9]]
print(t)