import random
import logging
import argparse
import tensorflow as tf
from model import DRQA
from train import batch_generator, Score, load_data


def main():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('--path', default='./best_checkpoints/', help='Path to model weights (tf checkpoint)')
    parser.add_argument('--verbosity', type=int, default=0, help='set 1 to show 10 samples')
    args = parser.parse_args()

    options = vars(args)

    extra_options = {
        'grad_clipping': 10.0,
        'learning_rate': 0.01,
        'layers_num': 2,
        'hidden_size': 128,
        'dropout_rate': 0.7,
        'extra_data': './extra_data/',
        'max_len': 15,
        'pos_num': 50,
        'pos_dim': 12,
        'ner_num': 19,
        'ner_dim': 8,
        'thresh_tune': 1000,
    }

    options_copy = options.copy()
    options_copy.update(extra_options)

    options = options_copy

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)

    log.info('Logging initialized')
    dev, dev_answers, embeddings, new_options = load_data(options, test_mode=True)
    options = new_options
    log.info('Data loaded')

    graph = tf.Graph()
    with graph.as_default():
        log.info('Loading graph')
        model = DRQA(options, embeddings)
        log.info('Graph loaded')

        save_path = tf.train.latest_checkpoint(options['path'])
        saver = tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess, save_path)

            log.info('Validation in process...')
            predictions = []
            for batch in batch_generator(dev, options['batch_size'], options, test_mode=True):
                predictions.extend(model.test(batch, sess))

            if options['verbosity'] == 1:
                idx = random.sample(range(len(predictions)), 10)
                for p, t in zip([predictions[i] for i in idx], [dev_answers[i] for i in idx]):
                    print('prediction={}\ntruth={}'.format(p, t))
                    print('-' * 30)

            score = Score(predictions, dev_answers)
            f1_score = score.computef1()
            log.info('Validation completed!')
            log.info("dev F1: {}".format(f1_score))


if __name__ == '__main__':
    main()