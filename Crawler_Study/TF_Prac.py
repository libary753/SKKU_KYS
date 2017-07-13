import tensorflow as tf

a = tf.constant(5, name = "input_a")
b = tf.constant(3, name = "input_b")
c = tf.add(a,b, name = "add_c")
d = tf.multiply(a,b, name = "mul_d")
e = tf.add(c,d, name = "add_e")

sess = tf.Session()
sess.run(e)
writer = tf.summary.FileWriter("./new_graph",sess.graph)
sess.close()