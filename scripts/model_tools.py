import tensorflow as tf

class ModelTools():
    def initialize_weights(self, shape):
        '''
        '''
        return tf.Variable(tf.random_normal_initializer(stddev=0.25)(shape=shape))

    def add_biases(self, shape, value=0.05):
        return tf.Variable(tf.constant(value=value, shape=shape))


    def conv_layer(self, layer, kernel, input_channels, output_channels, stride_size, padding='SAME'):
        '''

        '''
        weights = self.initialize_weights([kernel, kernel, input_channels, output_channels])
        biases = self.add_biases([output_channels])
        # stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1]
        stride = [1, stride_size, stride_size, 1]
        # does a convolution scan on the given image
        layer = tf.nn.conv2d(layer, weights, strides=stride, padding=padding) + biases
        return layer

    def pool_layer(self, layer, kernel, stride):
        kernel_size = [1, kernel, kernel, 1]
        stride_size = [1, stride, stride, 1]
        return tf.nn.max_pool(layer, ksize=kernel_size, strides=stride_size, padding="VALID")

    def flattening_layer(self, layer):
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]), new_size

    def fully_connected_layer(self, layer, input_shape, output_shape):
        weights = self.initialize_weights([input_shape, output_shape])
        biases = self.add_biases([output_shape])
        layer = tf.matmul(layer, weights) + biases  # mX+b
        return layer

    def activation_layer(self, layer):
        return tf.nn.relu(layer)
