import numpy as np
import tenosrflow as tf

from tenosrflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Defining hyper parameters
parameters = {
	"BACH_SIZE": 100,
	"HIDDEN_DIMS": 7,
	"RECURRENT_LENGTH": 7,
	"OUT_HIDDEN_DIMS": 32,
	"OUT_RECURRENT_LENGTH": 2,

	"LEARNING_RATE": 1e-3,
	"EPOCH": 20,
}

# preparing the datasets
next_train_batch = lambda x: mnist.train.next_batch(x)[0]
next_test_batch = lambda x: mnist.test.next_batch(x)[0]


