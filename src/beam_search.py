from nn import *
from utils import strip_eos
from copy import deepcopy

class BeamState(object):
    def __init__(self, h, inp, sent, nll):
        self.h, self.inp, self.sent, self.nll = h, inp, sent, nll

class Decoder(object):

    def __init__(self, sess, args, vocab, model):
        dim_h = args.dim_y + args.dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        self.vocab = vocab
        self.model = model
        self.max_len = args.max_seq_length
        self.beam_width = args.beam
        self.sess = sess

        cell = create_cell(dim_h, n_layers, dropout=1)

        self.inp = tf.placeholder(tf.int32, [None])
        self.h = tf.placeholder(tf.float32, [None, dim_h])

        tf.get_variable_scope().reuse_variables()
        embedding = tf.get_variable('embedding', [vocab.size, dim_emb])
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        with tf.variable_scope('generator'):
            inp = tf.nn.embedding_lookup(embedding, self.inp)
            outputs, self.h_prime = cell(inp, self.h)
            logits = tf.matmul(outputs, proj_W) + proj_b
            log_lh = tf.log(tf.nn.softmax(logits))
            self.log_lh, self.indices = tf.nn.top_k(log_lh, self.beam_width)

    def decode(self, h):
        go = self.vocab.word2id['<go>']
        batch_size = len(h)
        init_state = BeamState(h, [go] * batch_size,
            [[] for i in range(batch_size)], [0] * batch_size)
        beam = [init_state]

        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = self.sess.run(
                    [self.log_lh, self.indices, self.h_prime],
                    feed_dict={self.inp: state.inp, self.h: state.h})
                for i in range(batch_size):
                    for l in range(self.beam_width):
                        exp[i].append(BeamState(h[i], indices[i,l],
                            state.sent[i] + [indices[i,l]],
                            state.nll[i] - log_lh[i,l]))

            beam = [deepcopy(init_state) for _ in range(self.beam_width)]
            for i in range(batch_size):
                a = sorted(exp[i], key=lambda k: k.nll)
                for k in range(self.beam_width):
                    beam[k].h[i] = a[k].h
                    beam[k].inp[i] = a[k].inp
                    beam[k].sent[i] = a[k].sent
                    beam[k].nll[i] = a[k].nll

        return beam[0].sent

    def rewrite(self, batch):
        """
        Takes a batch of sentences (each encoded as a list of vocab ids), creates a latent representation for each in
        both domains, and the decodes those latent representations, effectively reconstructing the sentences in original
        and target domains
        :param batch: input batch, a dictionary as returned by utils.get_batch
        :return: ori, tsf. ori is a list of sentences (i.e. list of words as str) reconstructed in the original domain
        from their latent representation. tsf has the same format as ori but the sentences are reconstructed from their
        transferred latent representation
        """
        model = self.model
        h_ori, h_tsf= self.sess.run([model.h_ori, model.h_tsf],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.labels: batch['labels']})
        ori = self.decode(h_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf = self.decode(h_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf

    def rewrite_experiment(self, batch, styler_vec):
        """
        Encondes and reconsturcts a given batch of sentences just like rewrite, but using the experiment styler_vec
        technique, which consists on adding a styler vector to the latent representation of a sentence in its original
        domain, and decoding to its original domain itself. Due to applying the styler vector to the latent encoding,
        it is expected that decoded sentence is nevertheless, transferred, even though it was decoded using the original
        encoding
        :param batch: input batch, a dictionary as returned by utils.get_batch
        :param styler_vec: a vector to add to the original encoded sentences to alter its decoding and bias it towards
        the target domain
        :return: ori, tsf. ori is a list of sentences (i.e. list of words as str) reconstructed in the original domain
        from their latent representation. tsf has the same format as ori but the sentences are reconstructed from their
        transferred latent representation
        """
        model = self.model
        # h_ori is the encoder z state concatenated with the original y label latent representation
        # h_tsf is exactly the same, but instead of generating the y latent representation with the labels, it
        # generates it with 1 - labels, so h_tsf is z + other_y
        h_ori, h_tsf= self.sess.run([model.h_ori, model.h_tsf],
                                    feed_dict={model.dropout: 1,
                                    model.batch_size: batch['size'],
                                    model.enc_inputs: batch['enc_inputs'],
                                    model.labels: batch['labels']})
        h_tsf_experiment = h_ori + styler_vec
        ori = self.decode(h_ori)
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf_experiment = self.decode(h_tsf_experiment)
        tsf_experiment = [[self.vocab.id2word[i] for i in sent] for sent in tsf_experiment]
        tsf_experiment = strip_eos(tsf_experiment)

        tsf = self.decode(h_tsf)
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf, tsf_experiment
