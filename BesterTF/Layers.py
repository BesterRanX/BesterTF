import tensorflow as tf


class Layer():
    def __init__(self, output_dim, input_dim=0, activation=None):
        # cache parameters
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim



class Dense(Layer):
    def __init__(self, output_dim, input_dim=0, activation=None):
        # super class init
        Layer.__init__(output_dim, input_dim, activation)

    def compile(self):
        # initialise weights
        self.Weights = tf.Variable(tf.random_uniform([self.input_dim, self.output_dim], -1, 1))
        # initialise biases
        self.biases = tf.Variable(tf.zeros([1, self.output_dim]) + 0.1)

    # activation
    def act(self, inputs=None):
        Wx_plus_b = tf.matmul(inputs, self.Weights, name='Wx_plus_b') + self.biases
        return self.activation(Wx_plus_b)
