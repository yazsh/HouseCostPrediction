import tensorflow as tf

from HouseCostCleanData import *


learning_rate = .01
input_features = 288
hidden1 = 25
hidden2 = 25
hidden3 = 25
hidden4 = 1

weights = dict(w1=tf.Variable(tf.random_normal([input_features, hidden1])),
               w2=tf.Variable(tf.random_normal([hidden1, hidden2])),
               w3=tf.Variable(tf.random_normal([hidden2, hidden3])),
               w4=tf.Variable(tf.random_normal([hidden3, hidden4])))

biases = dict(b1=tf.Variable(tf.random_normal([hidden1])),
               b2=tf.Variable(tf.random_normal([hidden2])),
               b3=tf.Variable(tf.random_normal([hidden3])),
               b4=tf.Variable(tf.random_normal([hidden4])))

x = tf.placeholder("float32", [None, input_features], "x")
y = tf.placeholder("float32",[None, 1], "y")


layer = create_layer(x, biases['b1'], weights['w1'],tf.nn.relu);
layer = create_layer(layer, biases['b2'], weights['w2'],tf.nn.relu);
# layer = tf.nn.dropout(layer,.99)
layer = create_layer(layer, biases['b3'], weights['w3'],tf.nn.relu);

Z4 = create_layer(layer, biases['b4'], weights['w4'], tf.nn.relu);

# cost = tf.losses.mean_squared_error(labels = y, predictions=Z4)
cost = tf.reduce_mean(tf.sqrt(tf.square(tf.losses.absolute_difference(y, Z4))))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

end = 600
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for iteration in range(1, end):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        if iteration == (end - 1):
            print("Iteration " + str(iteration) + " cost: " + str(c))

    accuracy = tf.reduce_mean(tf.sqrt(tf.square(tf.losses.absolute_difference(y, Z4))))
    prediction = sess.run(Z4, feed_dict={x: train_features, y: train_labels})

    #dev prediction:
    dev_accuracy = sess.run(accuracy, feed_dict={x: dev_features, y: dev_labels})
    print(dev_accuracy)
    #test prediction
    test_prediction = sess.run(Z4, feed_dict={x: test_features, y:train_labels})
    test_prediction = pd.DataFrame(test_prediction)
    test_prediction.index += 1461
    test_prediction.to_csv("/Users/yazen/Desktop/datasets/HouseCostData/prediction.csv")
