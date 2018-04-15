import tensorflow as tf
import numpy as np


def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)


def create_cell(dim, n_layers, dropout):
    """
    Creates a GRU cell RNN with dropout and stacks it into a multilayer RNN if more than 1 layers are required
    :param dim: dimension of the input state
    :param n_layers: number of stacked layers
    :param dropout: dropout rate
    :return: a GRU cell RNN with dropout with input of dim dim
    """
    cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
    if n_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    return cell


def retrive_var(scopes):
    """
    Retrieves variables from different scopes in the graph
    :param scopes: the scopes to search in
    :return: list of variables
    """
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 scope=scope)
    return var


def linear(inp, dim_out, scope, reuse=False):
    """
    Transforms a [a x b] input matrix into a [a x dim_out] matrix
    through a fully-connected linear NN layer trained in the given scope
    :param inp: input matrix
    :param dim_out: output dimension
    :param scope: variable scope indicator
    :param reuse: whether or not to reuse the variable names in the scope. Default is False.
    :return: a transformation of inp into a [a x dim_out] matrix
    """
    dim_in = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.get_variable('W', [dim_in, dim_out])
        b = tf.get_variable('b', [dim_out])
    return tf.matmul(inp, W) + b


def combine(x, y, scope, reuse=False):
    dim_x = x.get_shape().as_list()[-1]
    dim_y = y.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.get_variable('W', [dim_x + dim_y, dim_x])
        b = tf.get_variable('b', [dim_x])

    h = tf.matmul(tf.concat(axis=1, values=[x, y]), W) + b
    return leaky_relu(h)


def feed_forward(inp, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W1 = tf.get_variable('W1', [dim, dim])
        b1 = tf.get_variable('b1', [dim])
        W2 = tf.get_variable('W2', [dim, 1])
        b2 = tf.get_variable('b2', [1])
    h1 = leaky_relu(tf.matmul(inp, W1) + b1)
    logits = tf.matmul(h1, W2) + b2

    return tf.reshape(logits, [-1])


def gumbel_softmax(logits, gamma, eps=1e-20):
    U = tf.random_uniform(tf.shape(logits))
    G = -tf.log(-tf.log(U + eps) + eps)
    return tf.nn.softmax((logits + G) / gamma)


def softsample_word(dropout, proj_W, proj_b, embedding, gamma):
    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func


def softmax_word(dropout, proj_W, proj_b, embedding, gamma):
    """
    Returns the softmax function for a given output
    The softmax returns the inp and logits
    :param dropout: dropout rate
    :param proj_W: projection matrix W
    :param proj_b: projection bias b
    :param embedding: [vocab size x dim_emb] embedding matrix
    :param gamma: temperature. The higher, the more skewed the distribution becomes
    :return: the softmax function for a given output.
    """
    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = tf.nn.softmax(logits / gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func


def argmax_word(dropout, proj_W, proj_b, embedding):
    """
    Returns the argmax function for a given output
    The argmax returns the inp and logits
    :param dropout: dropout rate
    :param proj_W: projection matrix W
    :param proj_b: projection bias b
    :param embedding: [vocab size x dim_emb] embedding matrix
    :return: the argmax function for a given output.
    """
    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        word = tf.argmax(logits, axis=1)
        inp = tf.nn.embedding_lookup(embedding, word)
        return inp, logits

    return loop_func


def rnn_decode(h, inp, length, cell, loop_func, scope):
    """
    Calculates the softmax or argmax over the sequence of hidden states
    :param h: hidden state
    :param inp: input token
    :param length: maximum sequence length
    :param cell: RNN cell
    :param loop_func: Loop function (softmax or argmax)
    :param scope: variable scope
    :return: sequences of hidden states and their logits based on the loop function: h_seq, logits_seq
    """
    h_seq, logits_seq = [], []

    with tf.variable_scope(scope):
        tf.get_variable_scope().reuse_variables()
        for _ in range(length):
            h_seq.append(tf.expand_dims(h, 1))
            output, h = cell(inp, h)
            inp, logits = loop_func(output)
            logits_seq.append(tf.expand_dims(logits, 1))

    # the lists get converted into a single tensor with each time step concatenated on the columns
    return tf.concat(axis=1, values=h_seq), tf.concat(axis=1, values=logits_seq)


def cnn(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, axis=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(axis=1, values=outputs)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output'):
            W = tf.get_variable('W', [n_filters * len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])

    return logits


def discriminator(x_real, x_fake, ones, zeros,
                  filter_sizes, n_filters, dropout, scope,
                  wgan=False, eta=10):
    d_real = cnn(x_real, filter_sizes, n_filters, dropout, scope)
    d_fake = cnn(x_fake, filter_sizes, n_filters, dropout, scope, reuse=True)

    if wgan:
        eps = tf.random_uniform([], 0.0, 1.0)
        mix = eps * x_real + (1 - eps) * x_fake
        d_mix = cnn(mix, filter_sizes, n_filters, dropout, scope, reuse=True)
        grad = tf.gradients(d_mix, mix)[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2]))
        loss = d_fake - d_real + eta * tf.square(grad_norm - 1)
        return tf.reduce_mean(loss)

    else:
        loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=ones, logits=d_real))
        loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=zeros, logits=d_fake))
        return loss_real + loss_fake
