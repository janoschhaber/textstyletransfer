import sys
import argparse
import pprint
import tensorflow as tf

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train',
            type=str,
            default='')
    argparser.add_argument('--dev',
            type=str,
            default='')
    argparser.add_argument('--test',
            type=str,
            default='')
    argparser.add_argument('--online_testing',
            type=bool,
            default=False)
    argparser.add_argument('--output',
            type=str,
            default='')
    argparser.add_argument('--vocab',
            type=str,
            default='')
    argparser.add_argument('--model',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            type=bool,
            default=False)

    argparser.add_argument('--batch_size',
            type=int,
            default=64) # default = 64
    argparser.add_argument('--max_epochs',
            type=int,
            default=20) # default = 20
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=1000) # default = 1000
    argparser.add_argument('--max_seq_length',
            type=int,
            default=20)
    argparser.add_argument('--max_train_size',
            type=int,
            default=-1)

    argparser.add_argument('--beam',
            type=int,
            default=8)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.7)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=200)
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0001)
    #argparser.add_argument('--learning_rate_decay',
    #        type=float,
    #        default=0.5)
    argparser.add_argument('--rho',                     # loss_g - rho * loss_d
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',              # softmax(logit / gamma)
            type=float,
            default=1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=0.5)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.001)
    argparser.add_argument('--max_filter_width',
            type=int,
            default=4)
    argparser.add_argument('--n_filters',
            type=int,
            default=256)

    args = argparser.parse_args()

    tf.logging.info('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    tf.logging.info('------------------------------------------------')

    return args


def load_arguments_live():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument('--train',
                           type=str,
                           default='')
    argparser.add_argument('--dev',
                           type=str,
                           default='')
    argparser.add_argument('--test',
                           type=str,
                           default='')

    argparser.add_argument('--online_testing',
            type=bool,
            default=True)
    argparser.add_argument('--output',
            type=str,
            default='/media/janosch/DATA/Documents/Mostafa/textstyletransfer/textstyletransfer/tmp/sentiment.test')
    argparser.add_argument('--vocab',
            type=str,
            default='/media/janosch/DATA/Documents/Mostafa/textstyletransfer/textstyletransfer/tmp/yelp.vocab')
    argparser.add_argument('--model',
            type=str,
            default='/media/janosch/DATA/Documents/Mostafa/textstyletransfer/textstyletransfer/tmp/model')
    argparser.add_argument('--load_model',
            type=bool,
            default=True)

    argparser.add_argument('--batch_size',
                           type=int,
                           default=64)  # default = 64
    argparser.add_argument('--max_epochs',
                           type=int,
                           default=1)  # default = 20
    argparser.add_argument('--steps_per_checkpoint',
                           type=int,
                           default=1000)  # default = 1000
    argparser.add_argument('--max_seq_length',
                           type=int,
                           default=20)
    argparser.add_argument('--max_train_size',
                           type=int,
                           default=-1)

    argparser.add_argument('--beam',
            type=int,
            default=8)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.7)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=200)
    argparser.add_argument('--dim_z',
            type=int,
            default=500)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.0001)
    #argparser.add_argument('--learning_rate_decay',
    #        type=float,
    #        default=0.5)
    argparser.add_argument('--rho',                     # loss_g - rho * loss_d
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',              # softmax(logit / gamma)
            type=float,
            default=1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=0.5)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.001)
    argparser.add_argument('--max_filter_width',
            type=int,
            default=4)
    argparser.add_argument('--n_filters',
            type=int,
            default=256)

    args = argparser.parse_args()

    tf.logging.info('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    tf.logging.info('------------------------------------------------')

    return args
