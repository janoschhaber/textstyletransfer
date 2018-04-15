import tensorflow as tf


def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
            for sent in sents]


def feed_dictionary(model, batch, rho, gamma, dropout=1, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.rho: rho,
                 model.gamma: gamma,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.dec_inputs: batch['dec_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights'],
                 model.labels: batch['labels']}
    return feed_dict


def makeup(_x, n):
    """
    Increase the length of _x to be n by duplicating some elements
    :param _x: input length
    :param n: output size
    :return: list of length n with all original and additional duplicate elements from _x
    """
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x


def reorder(order, _x):
    """
    Reorder a list based on a given ordering
    :param order: ordering key
    :param _x: list to reorder
    :return: reordered version of list _x
    """
    x = range(len(_x))
    for i, a in zip(order, _x):
        x[i] = a
    return x


def get_batch(x, labels, word2id, min_len=5):
    """
    Generates a batch from the given input
    :param x:
    :param labels:
    :param word2id:
    :param min_len:
    :return: dictionary with the batch statistics
    """
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    rev_x, go_x, x_eos, weights = [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        # rev_x = padding plus reversed input x
        rev_x.append(padding + sent_id[::-1])
        # go_x = go_token plus input x plus padding to max sentence length
        go_x.append([go] + sent_id + padding)
        # x_eos = sentence plus eos token plus padding to max sentence length
        x_eos.append(sent_id + [eos] + padding)
        # weights = 1 for actual words and 0 for others
        weights.append([1.0] * (l + 1) + [0.0] * (max_len - l))

    return {'enc_inputs': rev_x,
            'dec_inputs': go_x,
            'targets': x_eos,
            'weights': weights,
            'labels': labels,
            'size': len(x),
            'len': max_len + 1}


def get_batches(x0, x1, word2id, batch_size):
    # Make the data sets equally big by duplicating some elements of the shorter one
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
    # Determine size of the data sets
    n = len(x0)

    # Some reordering.
    order0 = range(n)
    # list of tuples (index, sentence), sorted by sentence length. This ensures batches have similar length to minimize
    # average padding per batch and not waste computation on 0s
    sorted_zip = sorted(zip(order0, x0), key=lambda i: len(i[1]))
    order0, x0 = zip(*sorted_zip)

    order1 = range(n)
    sorted_zip = sorted(zip(order1, x1), key=lambda i: len(i[1]))
    order1, x1 = zip(*sorted_zip)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        # Generate a new batch from the input data. Each batch has half x0, half x1
        batches.append(get_batch(x0[s:t] + x1[s:t], [0] * (t - s) + [1] * (t - s), word2id))
        s = t

    return batches, order0, order1


def get_batches_single_domain(x, word2id, batch_size, domain):
    """
    Creates batches of a single domain
    :param x: list of examples. List[List[str]]
    :param word2id: dictionary with the id of each word
    :param batch_size: size of the batch
    :param domain: specifies the domain, Union[0, 1]
    :return: batches, order, where batches is a list of batches, each one a dictionary as returned by get_batch. order
    is a List[int] indicating the order of the sentences in the batches, as they are sorted by length
    """
    n = len(x)
    order = range(n)
    sorted_zip = sorted(zip(order, x), key=lambda i: len(i[1]))
    order, x = zip(*sorted_zip)
    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(x[s:t], [domain] * (t - s), word2id))
        s = t
    return batches, order


def feed_dict_experiment(model, batch, args):
    """
    Gets the feed dict for Fab's experiment
    :return: a feed dict
    """
    return {
            model.dropout: 1,
            model.batch_size: batch['size'],
            model.enc_inputs: batch['enc_inputs'],
            model.dec_inputs: batch['dec_inputs'],
            model.labels: batch['labels'],
            model.gamma: args.gamma_min
    }


def get_config():
    """
  Returns config for tf.session
  """
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config
