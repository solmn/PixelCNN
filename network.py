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

height, width, channel = 28, 28, 1

# Masked convolutional 2d

def masked_conv2d(inputs, num_outputs, kernel_shape, mask_type, strides = [1,1], padding="SAME", activation_function scope = "masked_conv2d"):
	with tf.variable_scope(scope):
		mask_type mask_type.lower()
		batch_size, height, width, channel = inputs.get_shape().as_list()

		weight_h, weight_w = kernel_shape

		center_h = weight_h // 2
		center_w = weight_w // 2
		weight_shape = (weight_h, weight_w, channel, num_outputs)
		weights = tf.get_variable("weights", weight_shape, tf.float32, weights_initializer=tf.contrib.layers.xavier_initializer)
		if mask_type is not None:
			mask = np.ones((weight_h, weight_w, channel, num_outputs), dtype=np.float32)
			mask[center_h, center_w + 1:, :,:] = 0
			mask[center_h:, :, :, :] = 0
			if(mask_type == 'a'):
				mask[center_h, center_w, :, :] = 0
			weights *= tf.constant(mask, dtype=tf.float32)
		outputs = tf.nn.conv2d(inputs, weights, [1, strides[0], strides[1], 1], padding=padding, name='outputs')

		biases = tf.get_variable("biases", [num_outputs], tf.float32, biases_initializer=tf.zeros_initializer)
		outputs = tf.nn.bias_add(outputs, biases, name="output_plus_bias")
		outputs = activation_function(outputs, name="output_with_activation_fucntion")
		return outputs








