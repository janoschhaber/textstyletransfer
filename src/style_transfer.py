import os
import sys
import time
import random

from vocab import Vocabulary, build_vocab
from losses import Losses
from options import load_arguments
from file_io import load_sent, write_sent
from utils import *
from nn import *
from greedy_decoding import Decoder
from utils import get_config
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


class Model(object):
    def __init__(self, args, vocab, logdir):
        dim_y = args.dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = range(1, 1 + args.max_filter_width)
        n_filters = args.n_filters

        with tf.name_scope("network_parameters"):
            self.dropout = tf.placeholder(tf.float32, name='dropout')
            self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
            self.rho = tf.placeholder(tf.float32, name='rho')
            self.gamma = tf.placeholder(tf.float32, name='gamma')

        with tf.name_scope("batch_properties"):
            self.batch_len = tf.placeholder(tf.int32, name='batch_len')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')

        with tf.name_scope("inputs"):
            self.enc_inputs = tf.placeholder(tf.int32, [None, None], name='enc_inputs')
            self.dec_inputs = tf.placeholder(tf.int32, [None, None], name='dec_inputs')
            self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
            self.weights = tf.placeholder(tf.float32, [None, None], name='weights')
            self.labels = tf.placeholder(tf.float32, [None], name='labels')

            labels = tf.reshape(self.labels, [-1, 1])

        with tf.name_scope("embeddings"):
            embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
            if args.train: _add_emb_vis(embedding, logdir)

            with tf.variable_scope('projection'):
                proj_W = tf.get_variable('W', [dim_h, vocab.size])
                proj_b = tf.get_variable('b', [vocab.size])

            enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
            dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        with tf.name_scope("auto-encoder"):
            init_state = tf.concat(axis=1, values=[linear(labels, dim_y, scope='encoder'), tf.zeros([self.batch_size, dim_z])])
            cell_e = create_cell(dim_h, n_layers, self.dropout)
            _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs, initial_state=init_state, scope='encoder')
            z = z[:, dim_y:]

        with tf.name_scope("generator"):
            self.h_ori = tf.concat(axis=1, values=[linear(labels, dim_y, scope='generator'), z])
            self.h_tsf = tf.concat(axis=1, values=[linear(1 - labels, dim_y, scope='generator', reuse=True), z])

            cell_g = create_cell(dim_h, n_layers, self.dropout)
            g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs, initial_state=self.h_ori, scope='generator')

            # attach h0 in the front
            teach_h = tf.concat(axis=1, values=[tf.expand_dims(self.h_ori, 1), g_outputs])

            g_outputs = tf.nn.dropout(g_outputs, self.dropout)
            g_outputs = tf.reshape(g_outputs, [-1, dim_h])
            g_logits = tf.matmul(g_outputs, proj_W) + proj_b

            loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]), logits=g_logits)
            loss_g *= tf.reshape(self.weights, [-1])
            self.loss_g = tf.reduce_sum(loss_g) / tf.to_float(self.batch_size)
            tf.summary.scalar('loss_g', self.loss_g)

            #####   feed-previous decoding   #####
            go = dec_inputs[:, 0, :]
            soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,
                                        self.gamma)
            hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)

            soft_h_ori, soft_logits_ori = rnn_decode(self.h_ori, go, max_len, cell_g, soft_func, scope='generator')
            soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len, cell_g, soft_func, scope='generator')

            hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len, cell_g, hard_func, scope='generator')
            hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len, cell_g, hard_func, scope='generator')

        #####   discriminator   #####
        with tf.name_scope("discriminators"):
            # a batch's first half consists of sentences of one style,
            # and second half of the other
            half = tf.cast(self.batch_size/2, dtype=tf.int32)
            zeros, ones = self.labels[:half], self.labels[half:]
            soft_h_tsf = soft_h_tsf[:, :1 + self.batch_len, :]

            self.loss_d0 = discriminator(teach_h[:half], soft_h_tsf[half:],
                                         ones, zeros, filter_sizes, n_filters, self.dropout,
                                         scope='discriminator0')
            tf.summary.scalar('loss_d0', self.loss_d0)

            self.loss_d1 = discriminator(teach_h[half:], soft_h_tsf[:half],
                                         ones, zeros, filter_sizes, n_filters, self.dropout,
                                         scope='discriminator1')
            tf.summary.scalar('loss_d1', self.loss_d1)

        #####   optimizer   #####
        with tf.name_scope("optimizer"):
            self.loss_d = self.loss_d0 + self.loss_d1
            tf.summary.scalar('loss_d', self.loss_d)
            self.loss = self.loss_g - self.rho * self.loss_d
            tf.summary.scalar('loss', self.loss)

            theta_eg = retrive_var(['encoder', 'generator', 'embedding', 'projection'])
            theta_d0 = retrive_var(['discriminator0'])
            theta_d1 = retrive_var(['discriminator1'])

            self.optimizer_all = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.loss, var_list=theta_eg)
            self.optimizer_ae = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.loss_g, var_list=theta_eg)
            self.optimizer_d0 = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.loss_d0, var_list=theta_d0)
            self.optimizer_d1 = tf.train.AdamOptimizer(self.learning_rate) \
                .minimize(self.loss_d1, var_list=theta_d1)

        self.summary = tf.summary.merge_all()
        self.saver = tf.train.Saver()


def transfer(model, decoder, sess, args, vocab, data0, data1, out_path):
    batches, order0, order1 = get_batches(data0, data1, vocab.word2id, args.batch_size)

    data0_tsf, data1_tsf = [], []
    losses = Losses(len(batches))
    for batch in batches:
        ori, tsf = decoder.rewrite(batch)
        half = batch['size'] / 2
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]

        loss, loss_g, loss_d, loss_d0, loss_d1 = sess.run([model.loss, model.loss_g, model.loss_d, model.loss_d0, model.loss_d1],
                                                          feed_dict=feed_dictionary(model, batch, args.rho, args.gamma_min))
        losses.add(loss, loss_g, loss_d, loss_d0, loss_d1)

    n0, n1 = len(data0), len(data1)
    data0_tsf = reorder(order0, data0_tsf)[:n0]
    data1_tsf = reorder(order1, data1_tsf)[:n1]

    if out_path:
        write_sent(data0_tsf, out_path + '.0' + '.tsf')
        write_sent(data1_tsf, out_path + '.1' + '.tsf')

    return losses


def _add_emb_vis(embedding_var, logdir):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    summary_writer = tf.summary.FileWriter(logdir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = logdir + "/metadata.vocab"
    projector.visualize_embeddings(summary_writer, config)

def main(unused_argv):
    tf.logging.set_verbosity(
        tf.logging.INFO)  # choose what level of logging you want

    logdir = "../logs"
    args = load_arguments()
    tf.logging.info(tf.__version__)


    if args.train:
        #####   data preparation   #####
        train0 = load_sent(args.train + '.0', args.max_train_size)
        train1 = load_sent(args.train + '.1', args.max_train_size)
        tf.logging.info('#sents of training file 0: %d', len(train0))
        tf.logging.info('#sents of training file 1: %d', len(train1))
        #####

        if args.vocab:
            vocab_path = args.vocab
        else:
            vocab_path = os.path.dirname(args.train) + "/vocab.vocab"
            vocab_metadata_path = logdir + "/metadata.vocab"

        if not os.path.isfile(vocab_path):
            build_vocab(train0 + train1, vocab_path, vocab_metadata_path)

        vocab = Vocabulary(vocab_path)
        tf.logging.info('vocabulary size: %d', vocab.size)

        model = Model(args, vocab, logdir)

        saver = tf.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time
        sv = tf.train.Supervisor(logdir=logdir,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=60,
                                 # save summaries for tensorboard every 60 secs
                                 # save_model_secs=100,
                                 # checkpoint every 60 secs
                                 global_step=None
                                 )
        summary_writer = sv.summary_writer
        tf.logging.info("Preparing or waiting for session...")
        sess_context_manager = sv.prepare_or_wait_for_session(
            config=get_config())
        tf.logging.info("Created session.")

        with sess_context_manager as sess:

            if args.load_model:
                tf.logging.info('Loading model from', args.model)
                model.saver.restore(sess, args.model)

            batches, _, _ = get_batches(train0, train1, vocab.word2id, args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            losses = Losses(args.steps_per_checkpoint)
            best_dev = float('inf')
            learning_rate = args.learning_rate
            rho = args.rho
            gamma = args.gamma_init
            dropout = args.dropout_keep_prob

            tf.logging.info("Writing to file")

            for epoch in range(1, 1 + args.max_epochs):
                tf.logging.info('--------------------epoch %d--------------------' % epoch)
                tf.logging.info('learning_rate: %f, gamma:  %f' % (learning_rate, gamma))

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, rho, gamma, dropout, learning_rate)

                    loss_d0, _ = sess.run([model.loss_d0, model.optimizer_d0], feed_dict=feed_dict)
                    loss_d1, _ = sess.run([model.loss_d1, model.optimizer_d1], feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0 < 1.2 and loss_d1 < 1.2:
                        optimizer = model.optimizer_all
                    else:
                        optimizer = model.optimizer_ae

                    loss, loss_g, loss_d, summary, _ = \
                        sess.run([model.loss, model.loss_g,
                                  model.loss_d, model.summary, optimizer],
                                 feed_dict=feed_dict)

                    step += 1
                    summary_writer.add_summary(summary, step)

                    losses.add(loss, loss_g, loss_d, loss_d0, loss_d1)
                    if step % args.steps_per_checkpoint == 0:
                        losses.output('step %d, time %.0fs,' % (step, time.time() - start_time))
                        losses.clear()

                if args.dev:
                    #####   data preparation   #####
                    dev0 = load_sent(args.dev + '.0')
                    dev1 = load_sent(args.dev + '.1')

                    decoder = Decoder(sess, args, vocab, model)
                    dev_losses = transfer(model, decoder, sess, args, vocab, dev0, dev1, args.output + '.epoch%d' % epoch)
                    dev_losses.output('dev')
                    if dev_losses.loss < best_dev:
                        best_dev = dev_losses.loss
                        tf.logging.info('saving model...')
                        model.saver.save(sess, args.model)

                gamma = max(args.gamma_min, gamma * args.gamma_decay)
                summary_writer.flush()

    elif args.test:
        #####   data preparation   #####
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')
        vocab = Vocabulary(args.vocab)
        tf.logging.info('vocabulary size: %d', vocab.size)

        model = Model(args, vocab)
        sess = tf.Session(config=get_config())
        tf.logging.info('Loading model from', args.model)
        model.saver.restore(sess, args.model)
        decoder = Decoder(sess, args, vocab, model)
        test_losses = transfer(model, decoder, sess, args, vocab, test0, test1, args.output)
        test_losses.output('test')

    elif args.online_testing:
        vocab = Vocabulary(args.vocab)
        tf.logging.info('vocabulary size: %d', vocab.size)
        model = Model(args, vocab)
        sess = tf.Session(config=get_config())
        tf.logging.info('Loading model from', args.model)
        model.saver.restore(sess, args.model)
        decoder = Decoder(sess, args, vocab, model)
        while True:
            sys.stdout.write('> ')
            sys.stdout.flush()
            inp = sys.stdin.readline().rstrip()
            if inp == 'quit' or inp == 'exit':
                break
            inp = inp.split()
            y = int(inp[0])
            sent = inp[1:]

            batch = get_batch([sent], [y], vocab.word2id)
            ori, tsf = decoder.rewrite(batch)
            print('original:', ' '.join(w for w in ori[0]))
            print('transfer:', ' '.join(w for w in tsf[0]))



if __name__ == '__main__':
  tf.app.run()