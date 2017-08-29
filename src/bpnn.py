#!~/anaconda2/bin/python

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from tensorflow.examples.tutorials.mnist import input_data

model = Sequential()
# .add() # build a NN #
model.add(Dense(units=500, use_bias=False, activation='relu', input_shape=(784,)))
model.add(Dense(units=128,  use_bias=True,  activation='relu'))
model.add(Dense(units=10,  use_bias=True))
model.add(Activation("softmax")) ## Activation used like this, or inner Dense ##
# compile NN #
model.compile(loss='categorical_crossentropy', \
			  optimizer=SGD(lr=0.5, momentum=0.9, nesterov=False, decay=0.03), \
			  metrics=['accuracy'])
			  
# read train and test data #
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
train_data, test_data = mnist.train, mnist.test
x_test, y_test = test_data.images, test_data.labels
# train the model #
iter_nums  = 100000
batch_size = 100
for iter in range(iter_nums):
	x_batch, y_batch = train_data.next_batch(batch_size)
	model.train_on_batch(x_batch, y_batch)
	if iter%100==0:
		loss_and_metrics = model.evaluate(x_test, y_test)
		#classes = model.predict(x_test)
		print 'iter: ', iter, loss_and_metrics


