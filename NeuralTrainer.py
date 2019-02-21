import tensorflow as tf
import numpy as np


class NeuralTrainer:
    # Encapsulates a neural net trainer
    def __init__(self, inputNodes, outputNodes):
        # Useful numbers
        self.numHiddenNode1 = 256
        self.numHiddenNode2 = 256
        self.numInputNodes = inputNodes
        self.numOutputNodes = outputNodes
        self.epoch = 1
        self.learnRate = 0.00001
        self.ops = {}

    def InitialiseSession(self):
        self.sess = tf.Session()
        # graph inputs
        self.x_inputs = tf.placeholder(tf.float32, shape=[None, self.numInputNodes])
        self.y_label = tf.placeholder(tf.float32, shape=[None, self.numOutputNodes])

        # weights
        self.weightOne = tf.Variable(tf.random_normal([self.numInputNodes,
                                                  self.numHiddenNode1]))
        self.weightTwo = tf.Variable(tf.random_normal([self.numHiddenNode1,
                                                  self.numHiddenNode2]))
        self.weightOut = tf.Variable(tf.random_normal([self.numHiddenNode2,
                                                  self.numOutputNodes]))
        # biases
        self.biasOne = tf.Variable(tf.random_normal([self.numHiddenNode1]))
        self.biasTwo = tf.Variable(tf.random_normal([self.numHiddenNode2]))
        self.biasOut = tf.Variable(tf.random_normal([self.numOutputNodes]))

        # forward propagation
        self.predictionOne = tf.add(tf.matmul(self.x_inputs, self.weightOne), self.biasOne)
        self.predictionTwo = tf.add(tf.matmul(self.predictionOne, self.weightTwo), self.biasTwo)
        self.predOut = tf.matmul(self.predictionTwo, self.weightOut) + self.biasOut

        self.prediction = tf.nn.softmax(self.predOut)

        # backpropagation
        self.theLogits = tf.nn.softmax_cross_entropy_with_logits(logits=self.predOut, labels=self.y_label)
        self.loss = tf.reduce_mean(self.theLogits)
        self.optimiser = tf.train.AdamOptimizer(self.learnRate).minimize(self.loss)
        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.y_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # actual
        self.init = tf.global_variables_initializer()


        #
        # Save operators
        # self.ops["loss"] = loss
        # self.ops["optimiser"] = optimiser
        # self.ops["correct_prediction"] = correct_prediction
        # self.ops["accuracy"] = accuracy
        # self.ops["prediction"] = prediction
        # self.ops["init"] = init
        # self.x_inputs = x_inputs
        # self.y_label = y_label

    def TrainNet(self, imageset, epoch):
        x_train = imageset['data']
        y_train = imageset['labels']

        x_inputs = self.x_inputs
        y_label = self.y_label

        self.sess.run(self.init)

        for ii in range(epoch):
            self.sess.run(self.optimiser,
                          feed_dict={x_inputs: x_train, y_label: y_train})
            if ii % 100 == 0:
                # Analyse progress so far
                lossy = self.sess.run(self.loss,
                                      feed_dict={x_inputs: x_train,
                                                 y_label: y_train})
                acc = self.sess.run(self.accuracy,
                                    feed_dict={x_inputs: x_train,
                                               y_label: y_train})
                print('Training Step:' + str(ii) + ' out of ' +
                      str(epoch) + '  Accuracy =  ' + str(acc) +
                      '  Loss = ' + str(lossy))

        print("Training finished...")


    def SaveState(self, basename):

        saver = tf.train.Saver()
        save_path = saver.save(self.sess, basename)
        return save_path

    def RestoreState(self, metafile):

        tf.reset_default_graph()
        self.saver = tf.train.import_meta_graph(metafile)

    def TestNet(self, imageset, filepath):

        x_test = imageset['data']
        y_test = imageset['labels']

        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(filepath))

            check = sess.run([self.loss])
            print(check)

            check_loss = sess.run([self.loss], feed_dict={self.x_inputs:x_test, self.y_label:y_test})
            print("weight check: ", str(check_loss))
