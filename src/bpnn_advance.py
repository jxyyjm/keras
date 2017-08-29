#!~/anaconda2/bin/python

from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU, ThresholdedReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from tensorflow.examples.tutorials.mnist import input_data

model = Sequential()
# .add() # build a NN #
model.add(Dense(units=1000, use_bias=False, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, beta_initializer='zero')

model.add(Dense(units=600, use_bias=True,  activation=LeakyReLU(alpha=0.3), kernel_constraint=maxnorm(20.)))
model.add(Dropout(0.3))
BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, beta_initializer='zero')

model.add(Dense(units=128,  use_bias=True,  activation=PReLU(), kernel_constraint=maxnorm(25.)))
model.add(Dropout(0.4))
BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, beta_initializer='zero')

model.add(Dense(units=10,  use_bias=True))
model.add(Activation("softmax")) ## Activation used like this, or inner Dense ##
# compile NN #
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', \
#			  optimizer=SGD(lr=0.5, momentum=0.9, nesterov=False, decay=0.03), \
		      optimizer=Adagrad(lr=0.02, epsilon=1e-08, decay=0.0), \
			  metrics=['accuracy'])
			  
# read train and test data #
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
train_data, test_data = mnist.train, mnist.test
x_test, y_test = test_data.images, test_data.labels
# train the model #
iter_nums  = 100000
batch_size = 100
#model.fit(x_train, y_train, epochs=100, batch_size=100)
for iter in range(iter_nums):
	x_batch, y_batch = train_data.next_batch(batch_size)
	model.train_on_batch(x_batch, y_batch)
	if iter%100==0:
		loss_and_metrics = model.evaluate(x_test, y_test)
		#classes = model.predict(x_test)
		print 'iter: ', iter, loss_and_metrics
		#print classes[0:10, :]


