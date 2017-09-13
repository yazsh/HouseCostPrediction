import tensorflow as tf

from HouseCostCleanData import *


learning_rate = .1
input_features = 288
hidden1 = 20
hidden2 = 40
hidden3 = 60
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
layer = create_layer(layer, biases['b3'], weights['w3'],tf.nn.relu);
Z4 = create_layer(layer, biases['b4'], weights['w4']);

cost = tf.losses.mean_squared_error(labels = y, predictions=Z4)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    for iteration in range(1, 500):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_features, y: train_labels})
        print("Iteration " + str(iteration) + " cost: " + str(c))

    prediction = Z4
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Z4, y), "float"))
    prediction = sess.run(prediction, feed_dict={x: train_features, y: train_labels})
    print(np.append(prediction[:20], train_labels[:20],1))
    print(accuracy.eval({x: train_features, y: train_labels}))

    test_prediction = pd.DataFrame(prediction)
    test_prediction.to_csv("/Users/yazen/Desktop/datasets/HouseCostData/prediction.csv")
