import tensorflow as tf

class LogisticRegression:
    """A logistic regression model"""

    def __init__(self, X, num_classes):
        self.X = X
        self.num_classes = num_classes

        flattened = tf.layers.flatten(X)
        self.W = W = tf.Variable(tf.zeros((flattened.shape[1], num_classes)), name="W")
        self.b = b = tf.Variable(tf.zeros((num_classes)), name="b")

        self.logits = tf.matmul(flattened, W) + b

class DenseNet:
    """A multi-layer perceptron network"""

    def __init__(self, X, dropout_rate, num_classes):
        self.X = X
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        flattened = tf.layers.flatten(X)
        fc1 = tf.layers.dense(flattened, units=1000, activation=tf.nn.relu, name="fc1")
        fc2 = tf.layers.dense(fc1, units=500, activation=tf.nn.relu, name="fc2")
        fc3 = tf.layers.dense(fc2, units=500, activation=tf.nn.relu, name="fc3")
        dropout4 = tf.layers.dropout(fc3, rate=dropout_rate)
        fc5 = tf.layers.dense(dropout4, units=250, activation=tf.nn.relu, name="fc5")
        dropout6 = tf.layers.dropout(fc5, rate=dropout_rate)
        self.logits = tf.layers.dense(dropout6, units=num_classes, name="fc7")

class AlexNet:
    """A convolutional AlexNet network"""

    def __init__(self, X, dropout_rate, num_classes):
        self.X = X
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # 1st layer: Conv -> ReLU -> LRN -> MaxPool
        conv1 = tf.layers.conv2d(X, filters=96, kernel_size=(11, 11), strides=4, activation=tf.nn.relu, name="conv1")
        norm1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=1e-05, beta=0.75, bias=1.0, name="norm1")
        pool1 = tf.layers.max_pooling2d(norm1, pool_size=(3, 3), strides=2, name="pool1")

        # 2nd layer: Conv -> ReLU -> LRN -> MaxPool
        conv2 = tf.layers.conv2d(pool1, filters=256, kernel_size=(5, 5), padding="SAME", activation=tf.nn.relu, name="conv2")
        norm2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=1e-05, beta=0.75, bias=1.0, name="norm2")
        pool2 = tf.layers.max_pooling2d(norm2, pool_size=(3, 3), strides=2, name="pool2")

        # 3rd layer: Conv -> ReLU
        conv3 = tf.layers.conv2d(pool2, filters=384, kernel_size=(3, 3), padding="SAME", activation=tf.nn.relu, name="conv3")

        # 4th layer: Conv -> ReLU
        conv4 = tf.layers.conv2d(conv3, filters=384, kernel_size=(3, 3), padding="SAME", activation=tf.nn.relu, name="conv4")

        # 5th layer: Conv -> ReLU => MaxPool
        conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=(3, 3), padding="SAME", activation=tf.nn.relu, name="conv5")
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=(3, 3), strides=2, name="pool5")

        # 6th later: FC -> ReLU -> Dropout
        flattened = tf.layers.flatten(pool5)
        fc6 = tf.layers.dense(flattened, units=4096, activation=tf.nn.relu, name="fc6")
        dropout6 = tf.layers.dropout(fc6, rate=dropout_rate)

        # 7th layer: FC -> ReLU -> Dropout
        fc7 = tf.layers.dense(dropout6, units=4096, activation=tf.nn.relu, name="fc7")
        dropout7 = tf.layers.dropout(fc7, rate=dropout_rate)

        # 8th layer: FC -> unscaled activations
        self.logits = tf.layers.dense(dropout7, units=num_classes, name="fc8")

