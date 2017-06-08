import tensorflow as tf


class Sequential:
    def __init__(self):
        self.layers = []
        self.xs = tf.placeholder(tf.float32)
        self.ys = tf.placeholder(tf.float32)

    # add layers and cache them
    def add(self,Layer):
        self.layers.append(Layer)

    def compile(self, optimizer=None):
        # initialise dimensions
        for l in range(1, len(self.layers)):
            self.layers[l].input_dim = self.layers[l-1].output_dim

        # compile layers
        self.layers[0].compile()
        prediction = self.layers[0].act(self.xs) # activation

        for l in range(1, len(self.layers)):
            self.layers[l].compile()
            prediction = self.layers[l].act(prediction) # layer connected activation

        # updates
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.ys, prediction)))
        self.step = optimizer.minimize(self.loss)

    def fit(self, input_data=None, target_output=None, epochs=1000):
        init = tf.global_variables_initializer()
        # start to run
        with tf.Session() as sess:
            sess.run(init)

            for i in range(epochs):
                sess.run(self.step, feed_dict={self.xs:input_data, self.ys:target_output})
                # print loss each 50 loop
                if i % 50:
                    print("loss:", sess.run(self.loss, feed_dict={self.xs: input_data, self.ys:target_output}))