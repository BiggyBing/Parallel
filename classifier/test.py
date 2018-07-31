
#
#learning_rate = 0.0001
#epochs = 1
#batch_size = 50
#
#
#x = tf.placeholder(tf.float32, [None, 784])
## dynamically reshape the input
#x_shaped = tf.reshape(x, [-1, 28, 28, 1])
## now declare the output data placeholder - 10 digits
#y = tf.placeholder(tf.float32, [None, 10])
#
#layer1 = create_new_conv_layer(x_shaped,yf[:2], 1, 32, [5, 5], [2, 2], name='layer1')
#layer2 = create_new_conv_layer(layer1,yf[2:4], 32, 64, [5, 5], [2, 2], name='layer2')
#
#flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
#
## setup some weights and bias values for this layer, then activate with ReLU
#wd1 = tf.Variable(yf[4].reshape(7*7*64,1000), name='wd1')
#bd1 = tf.Variable(yf[5], name='bd1')
#dense_layer1 = tf.matmul(flattened, wd1) + bd1
#dense_layer1 = tf.nn.relu(dense_layer1)
#
## another layer with softmax activations
#wd2 = tf.Variable(yf[6].reshape(1000,10), name='wd2')
#bd2 = tf.Variable(yf[7], name='bd2')
#dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
#y_ = tf.nn.softmax(dense_layer2)
#
#
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_layer2, labels=y))
#
#optimizer = AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
## setup the initialisation operator
#init_op = tf.global_variables_initializer()
#grad = []
#with tf.Session() as sess:
#    # initialise the variables
#    sess.run(init_op)
#    total_batch = int(len(mnist.train.labels) / batch_size)
#    count = 0
#    for epoch in range(epochs):
#        avg_cost = 0
#        
#        for i in range(total_batch):
#            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
#            _,c = sess.run([optimizer,cross_entropy], feed_dict={x: batch_x, y: batch_y})
#            avg_cost += c / total_batch
#
#        test_acc = sess.run(accuracy, 
#                       feed_dict={x: mnist.test.images, y: mnist.test.labels})
#        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))
#
#    print("\nTraining complete!")
#    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
from sklearn import datasets
iris = datasets.load_iris()
train = iris.data
target = iris.target

#sending date to engines
