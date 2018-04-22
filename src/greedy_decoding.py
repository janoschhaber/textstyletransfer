import numpy as np
from utils import strip_eos

class Decoder(object):

    def __init__(self, sess, args, vocab, model):
        self.sess, self.vocab, self.model = sess, vocab, model

    def rewrite(self, batch):
        model = self.model
        logits_ori, logits_tsf = self.sess.run(
            [model.hard_logits_ori, model.hard_logits_tsf],
            feed_dict={model.dropout: 1,
                       model.batch_size: batch['size'],
                       model.enc_inputs: batch['enc_inputs'],
                       model.dec_inputs: batch['dec_inputs'],
                       model.labels: batch['labels']})

        ori = np.argmax(logits_ori, axis=2).tolist()
        ori = [[self.vocab.id2word[i] for i in sent] for sent in ori]
        ori = strip_eos(ori)

        tsf = np.argmax(logits_tsf, axis=2).tolist()
        tsf = [[self.vocab.id2word[i] for i in sent] for sent in tsf]
        tsf = strip_eos(tsf)

        return ori, tsf

    def rewrite_experiment(self, batch, styler_vec):
        """
        Rewrites a batch to the generator initial hidden states for the original and transferred domains
        :param batch: input batch
        :return: initial hidden states for the original and transferred domains
        """
        model = self.model
        h_ori, h_tsf = self.sess.run([model.h_ori, model.h_tsf],
                                     feed_dict={model.dropout: 1,
                                                model.batch_size: batch['size'],
                                                model.enc_inputs: batch['enc_inputs'],
                                                model.labels: batch['labels']})
        h_tsf_experiment = h_tsf + styler_vec
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