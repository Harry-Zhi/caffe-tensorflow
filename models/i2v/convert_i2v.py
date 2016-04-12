from .tensorflow import Network
import tensorflow as tf


class I2VNet(object):

    def __init__(self):
        self.net = I2V
        self.batch_size = 50
        self.n_channels = 3
        self.crop_size = 224


class I2V(Network):

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .max_pool(2, 2, 2, 2, name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .max_pool(2, 2, 2, 2, name='pool2')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .max_pool(2, 2, 2, 2, name='pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .max_pool(2, 2, 2, 2, name='pool5')
             .conv(3, 3, 1024, 1, 1, name='conv6_1')
             .conv(3, 3, 1024, 1, 1, name='conv6_2')
             .conv(3, 3, 1024, 1, 1, name='conv6_3')
             .fc(4096, relu=False, name='encode1')
             .sigmoid(name='encode1neuron'))


def save(path='./',
         fname='i2v.tfmodel',
         dst_nodes=['encode1/encode1']):
    g_1 = tf.Graph()
    with tf.Session(graph=g_1) as sess:
        x = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        net = I2VNet()
        sess.run(tf.initialize_all_variables())
        graph_def = tf.python.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, dst_nodes)
    g_2 = tf.Graph()
    with tf.Session(graph=g_2) as sess:
        tf.train.write_graph(
            tf.python.graph_util.extract_sub_graph(
                graph_def, dst_nodes), path, fname, as_text=False)


def get_model():
    x = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    net = I2VNet()
    model = net.net({'data': x})
    return {'net': model, 'x': x}
