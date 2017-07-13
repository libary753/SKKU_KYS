import tensorflow as tf

# data set
x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

# Hypothesis
hypothesis = x_train * w + b

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# minimize 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)


# Launch the graph in a session
sess = tf.Session()

# initialize 
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))

#------------------------------------------------------------------------#
# Use placeholder
w2 = tf.Variable(tf.random_normal([1]),name='weight')
b2 = tf.Variable(tf.random_normal([1]),name='bias')
x2 = tf.placeholder(tf.float32,shape=[None])
y2 = tf.placeholder(tf.float32,shape=[None])

hypothesis2 = x2 * w2 + b2
# cost / loss function
cost2 = tf.reduce_mean(tf.square(hypothesis2 - y2))

# minimize 
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train2 = optimizer2.minimize(cost2)

# Launch the graph in a session
sess2 = tf.Session()

sess2.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val,w_val,b_val, _ = \
        sess2.run([cost2,w2,b2,train2],feed_dict = {x2:[1,2,3,4,5], y2:[2,4,6,8,10]})
    if step %20 == 0:
        print(step, cost_val, w_val, b_val)