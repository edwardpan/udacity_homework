import tensorflow as tf

hello_constant = tf.constant("Hello World!")
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
	output = sess.run(y, feed_dict={x: "Test String", y: 111, z: 23.41})
	print(output)