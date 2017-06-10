# based on ideas from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf


class JamoCNN(object):
    """
    A CNN for text classification.
    based on the Character-level Convolutional Networks for Text Classification paper.
    """
    def __init__(self, num_classes=2, filter_sizes=(3,6,7,9,12,15,18), num_filters_per_size=(64,128,64,128,128,64,32), l2_reg_lambda=0.0, sequence_max_length=420, num_quantized_chars=52):

        # Placeholders for input, output
        self.input_x = tf.placeholder(tf.float32, [None, sequence_max_length, num_quantized_chars, 1], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # ================ Filter 1 ================
        with tf.variable_scope("conv-maxpool-1"):
            filter_shape = [filter_sizes[0], num_quantized_chars, 1, num_filters_per_size[0]]
            W = tf.get_variable("W", filter_shape, tf.float32, tf.truncated_normal_initializer(stddev=0.01))
            self.W1 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[0]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv1")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[0])//3 + 1
            pooled_1 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool1")

        # ================ Filter 2 ================
        with tf.name_scope("conv-maxpool-2"):
            filter_shape = [filter_sizes[1], num_quantized_chars, 1, num_filters_per_size[1]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
            self.W2 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[1]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv2")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[1])//3 + 1
            pooled_2 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool2")

        # ================ Filter 3 ================
        with tf.name_scope("conv-maxpool-3"):
            filter_shape = [filter_sizes[2], num_quantized_chars, 1, num_filters_per_size[2]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
            self.W3 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[2]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv3")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[2])//3 + 1
            pooled_3 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool3")

        # ================ Filter 4 ================
        with tf.name_scope("conv-maxpool-4"):
            filter_shape = [filter_sizes[3], num_quantized_chars, 1, num_filters_per_size[3]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
            self.W4 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[3]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv4")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[3])//3 + 1
            pooled_4 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool4")

        # ================ Filter 5 ================
        with tf.name_scope("conv-maxpool-5"):
            filter_shape = [filter_sizes[4], num_quantized_chars, 1, num_filters_per_size[4]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
            self.W5 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[4]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv5")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[4])//3 + 1
            pooled_5 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool5")

        # ================ Filter 6 ================
        with tf.name_scope("conv-maxpool-6"):
            filter_shape = [filter_sizes[5], num_quantized_chars, 1, num_filters_per_size[5]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
            self.W6 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[5]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv6")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[5])//3 + 1
            pooled_6 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool6")
            
        # ================ Filter 7 ================
        with tf.name_scope("conv-maxpool-7"):
            filter_shape = [filter_sizes[6], num_quantized_chars, 1, num_filters_per_size[6]]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.01), name="W")
            self.W7 = W
            b = tf.Variable(tf.truncated_normal(shape=[num_filters_per_size[6]], stddev=0.01), name="b")
            conv = tf.nn.conv2d(self.input_x, W, strides=[1, 3, 1, 1], padding="VALID", name="conv7")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            height = (sequence_max_length - filter_sizes[6])//3 + 1
            pooled_7 = tf.nn.max_pool(
                h,
                ksize=[1, height, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool7")
            
        # ================ Layer 7 ================
        with tf.name_scope("stack_max_pools") as scope:
            max_pool_1 = tf.reshape(pooled_1, shape = [-1, num_filters_per_size[0]])
            max_pool_2 = tf.reshape(pooled_2, shape = [-1, num_filters_per_size[1]])
            max_pool_3 = tf.reshape(pooled_3, shape = [-1, num_filters_per_size[2]])
            max_pool_4 = tf.reshape(pooled_4, shape = [-1, num_filters_per_size[3]])
            max_pool_5 = tf.reshape(pooled_5, shape = [-1, num_filters_per_size[4]])
            max_pool_6 = tf.reshape(pooled_6, shape = [-1, num_filters_per_size[5]])
            max_pool_7 = tf.reshape(pooled_7, shape = [-1, num_filters_per_size[6]])


            tf_max_pools = tf.concat([max_pool_1, max_pool_2, max_pool_3, max_pool_4, max_pool_5, max_pool_6, max_pool_7], axis = 1)
        
        num_features_total = num_filters_per_size[0] + num_filters_per_size[1] +num_filters_per_size[2] + num_filters_per_size[3] + num_filters_per_size[4] + num_filters_per_size[5]  + num_filters_per_size[6]

        # Fully connected layer 1
        with tf.name_scope("fc-1"):
            height = tf_max_pools.get_shape().as_list()[1] # layer_수
            weight = tf.Variable(tf.random_normal(shape = [height, 2]), dtype=tf.float32)
            bias = tf.Variable(tf.random_normal(shape = [2]), dtype = tf.float32)
            fc_output = tf.add(tf.matmul(tf_max_pools, weight), bias)
            predictions = tf.argmax(fc_output, 1, name="predictions")
            
        # ================ Loss and Accuracy ================
        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = fc_output, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")