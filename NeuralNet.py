import ImageHeader

import tensorflow as tf
import numpy as np
import re

class NeuralNet(object):
    imgHdr = ImageHeader.ImageHeader()
    x_train = np.empty(1)
    y_train = np.empty(1)
    x_test = np.empty(1)
    y_test = np.empty(1)

    def __init__(self, imageHdr):
        self.imgHdr = imageHdr

        self.x_train = np.empty((self.imgHdr.maxImages, self.imgHdr.imgHeight * self.imgHdr.imgWidth))
        self.y_train = np.empty(shape=(self.imgHdr.maxImages, 15))

        self.x_test = np.empty((self.imgHdr.maxImages, self.imgHdr.imgHeight * self.imgHdr.imgWidth))
        self.y_test = np.empty(shape=(self.imgHdr.maxImages, 15))

    def getTrainImg(self, index, filelabel, fileimg):

        # have encode the labels into one-hot arrays
        regex = re.compile(r'\d+')
        targetNum = regex.search(filelabel)
        num = int(targetNum.group())
        temp_encoder = np.zeros((1,15))
        temp_encoder[0, num-1] = 1
        self.y_train[index-1] = temp_encoder
        print("actual label: ", num, " y label check: ", self.y_train[index-1].argmax())


        #temp_train = np.empty(self.imgHdr.imgWidth *self.imgHdr.imgHeight)
        temp_train = fileimg
        self.x_train[index-1] = temp_train.flatten()
        print("train shape: ", self.x_train.shape)

        return self.x_train, self.y_train




    def getTestImg(self, index, filelabel, fileimg):

        regex = re.compile(r'\d+')
        targetNum = regex.search(filelabel)
        num = int(targetNum.group())
        #have encode the labels into one-hot arrays
        temp_encoder = np.zeros((1,15))
        temp_encoder[0, num-1] = 1
        print("test encoder: ", temp_encoder)
        self.y_test[index-1] = temp_encoder


        #temp_test  = np.empty(self.imgHdr.imgHeight*self.imgHdr.imgWidth)
        temp_test = fileimg
        self.x_test[index-1] = temp_test.flatten()
        print("test shape: ", self.x_test.shape)

        return self.x_test, self.y_test

    def trainMNIST(self):

        # inputs and labels
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test

        # network parameters
        lr = 0.00001
        epoch = 1650 * 5

        # nodes
        numHiddenNode1 = 256
        numHiddenNode2 = 256
        numInputNodes = self.imgHdr.imgWidth * self.imgHdr.imgHeight
        numOutputNodes = 15

        sess = tf.Session()

        # graph inputs
        x_inputs = tf.placeholder(tf.float32, shape=[None, numInputNodes])
        y_label = tf.placeholder(tf.float32, shape=[None, numOutputNodes])

        # weights
        weightOne = tf.Variable(tf.random_normal([numInputNodes, numHiddenNode1]))
        weightTwo = tf.Variable(tf.random_normal([numHiddenNode1, numHiddenNode2]))
        weightOut = tf.Variable(tf.random_normal([numHiddenNode2, numOutputNodes]))
        # biases
        biasOne = tf.Variable(tf.random_normal([numHiddenNode1]))
        biasTwo = tf.Variable(tf.random_normal([numHiddenNode2]))
        biasOut = tf.Variable(tf.random_normal([numOutputNodes]))

        # forward propagation
        predictionOne = tf.add(tf.matmul(x_inputs, weightOne), biasOne)
        predictionTwo = tf.add(tf.matmul(predictionOne, weightTwo), biasTwo)
        predictionOut = tf.matmul(predictionTwo, weightOut)+ biasOut

        prediction = tf.nn.softmax(predictionOut)

        #backpropagation
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictionOut, labels=y_label))
        optimiser = tf.train.AdamOptimizer(lr).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # actual
        init = tf.global_variables_initializer()
        sess.run(init)
        for ii in range(epoch):

            predictionCheck = sess.run(prediction, feed_dict={x_inputs: x_train})
            actualLabel = sess.run(y_label, feed_dict={y_label: y_train})
            print("prediction check: ", str(predictionCheck.argmax()), "  actual Label: ", actualLabel.argmax())

            # actual work being done
            sess.run(optimiser, feed_dict={x_inputs: x_train, y_label: y_train})

            if ii % 100 == 0:

                lossy = sess.run(loss, feed_dict={x_inputs:x_train, y_label:y_train})
                acc = sess.run(accuracy, feed_dict={x_inputs:x_train, y_label:y_train})
                print('Training Step:' + str(ii) + '  Accuracy =  ' + str(acc) + '  Loss = ' + str(lossy))

        print("Training finished...")

        predictionCheckTest = sess.run(prediction, feed_dict={x_inputs: x_test})
        actualLabelTest = sess.run(y_label, feed_dict={y_label: y_test})
        print("prediction check: ", str(predictionCheckTest.argmax()), "  actual Label: ", actualLabelTest.argmax())
        test_accuracy = sess.run(accuracy, feed_dict={x_inputs:x_test, y_label:y_test})
        print("Test Accuracy: ", test_accuracy)
        sess.close()