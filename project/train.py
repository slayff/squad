import re, os, sys
import random
import string
import logging
import argparse
from datetime import datetime
from collections import Counter
import tensorflow as tf
import numpy as np
import msgpack
import pandas as pd
from model import DRQA

def batch_generator(data, batch_size, options, test_mode=False):
    if not test_mode:
        indices = list(range(len(data)))
        random.shuffle(indices)
        data = [data[i] for i in indices]

    data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    for batch in data:
        batch_len = len(batch)
        batch = list(zip(*batch))

        context_len = options['context_len']
        question_len = options['question_len']
        features_num = options['features_num']

        context_id = np.zeros((batch_len, context_len))
        for i, doc in enumerate(batch[0]):
            context_id[i, :len(doc)] = doc

        context_feature = np.zeros((batch_len, context_len, features_num))
        for i, doc in enumerate(batch[1]):
            for j, feature in enumerate(doc):
                context_feature[i, j, :] = feature

        context_tag = np.zeros((batch_len, context_len))
        for i, doc_tags in enumerate(batch[2]):
            context_tag[i, :len(doc_tags)] = doc_tags

        context_ent = np.zeros((batch_len, context_len))
        for i, doc_ent in enumerate(batch[3]):
            context_ent[i, :len(doc_ent)] = doc_ent


        question_id = np.zeros((batch_len, question_len))
        for i, q_id in enumerate(batch[4]):
            question_id[i, :len(q_id)] = q_id

        context_mask = np.logical_not(np.equal(context_id, 0))
        question_mask = np.logical_not(np.equal(question_id, 0))

        if not test_mode:
            start_true = batch[5]
            end_true = batch[6]

        text = list(batch[-2])
        spans = list(batch[-1])
        if not test_mode:
            yield (context_id,
                   context_feature,
                   context_tag,
                   context_ent,
                   context_mask,
                   question_id,
                   question_mask,
                   start_true,
                   end_true,
                   text,
                   spans
                   )
        else:
            yield (context_id,
                   context_feature,
                   context_tag,
                   context_ent,
                   context_mask,
                   question_id,
                   question_mask,
                   text,
                   spans
                   )


def load_data(options, test_mode=False):
    def _get_max_len(lhs, rhs):
        return max(max(len(x) for x in lhs), max(len(x) for x in rhs))

    with open(options['extra_data']+'meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embeddings = meta['embedding']

    with open(options['extra_data']+'data.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    if not test_mode:
        train_ = pd.read_csv(options['extra_data']+'train.csv')

        train = list(zip(
            data['trn_context_ids'],
            data['trn_context_features'],
            data['trn_context_tags'],
            data['trn_context_ents'],
            data['trn_question_ids'],
            train_['answer_start_token'].tolist(),
            train_['answer_end_token'].tolist(),
            data['trn_context_text'],
            data['trn_context_spans']
        ))


    dev_ = pd.read_csv(options['extra_data'] + 'dev.csv')
    dev_answers = list(map(eval, dev_['answers'].values))
    dev = list(zip(
        data['dev_context_ids'],
        data['dev_context_features'],
        data['dev_context_tags'],
        data['dev_context_ents'],
        data['dev_question_ids'],
        data['dev_context_text'],
        data['dev_context_spans']
    ))

    options['context_len'] = _get_max_len(data['trn_context_ids'], data['dev_context_ids'])
    options['features_num'] = _get_max_len(data['trn_context_features'][0], data['dev_context_features'][0])
    options['question_len'] = _get_max_len(data['trn_question_ids'], data['dev_question_ids'])

    if not test_mode:
        return train, dev, dev_answers, embeddings, options
    else:
        return dev, dev_answers, embeddings, options

class Score:
    def __init__(self, predictions, correct):
        self.pred = predictions
        self.correct = correct

    def _normalize(self, text):
        def remove_articles(text_):
            text_ = re.sub(r'\b(a|an|the)\b', ' ', text_)
            return ' '.join(text_.split())

        def remove_punc(text_):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text_ if ch not in exclude)
        return remove_articles(remove_punc(text.lower()))

    def _computef1(self, prediction, answer_list):
        def _score(predicted, answer):
            common = Counter(predicted) & Counter(answer)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1. * num_same / len(predicted)
            recall = 1. * num_same / len(answer)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        pred_tokens = self._normalize(prediction).split()
        scores = [_score(pred_tokens, self._normalize(answer).split()) for answer in answer_list]
        return max(scores)

    def computef1(self):
        f1 = 0.
        for pred, corr in zip(self.pred, self.correct):
            f1 += self._computef1(pred, corr)

        return f1 / len(self.pred)



def main():

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-ln', '--layers_num', type=int, default=2)
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-d', '--dropout_rate', type=float, default=0.7)
    args = parser.parse_args()

    iter_check = '/iter_checkpoints/'
    best_check =  './best_checkpoints/'
    if not os.path.exists(iter_check):
        os.makedirs(iter_check, exist_ok=True)
    if not os.path.exists(best_check):
        os.makedirs(best_check, exist_ok=True)

    options = vars(args)

    extra_options = {
        'grad_clipping': 10.0,
        'learning_rate': 0.01,
        'extra_data': './extra_data/',
        'iter_check': './iter_checkpoints/',
        'best_check': './best_checkpoints/',
        'max_len': 15,
        'pos_num': 50,
        'pos_dim': 12,
        'ner_num': 19,
        'ner_dim': 8,
        'thresh_tune': 1000,
        'save_every_n_iter': 100,
        'dev_test_every_n_iter': 300,
    }

    options_copy = options.copy()
    options_copy.update(extra_options)

    options = options_copy

    log_file = './out.log'
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    log.info('Logging initialized')

    train, dev, dev_answers, embeddings, new_options = load_data(options)
    options = new_options
    log.info('Data loaded')

    graph = tf.Graph()
    with graph.as_default():
        log.info('Loading graph')
        model = DRQA(options, embeddings)
        log.info('Graph loaded')

        iter_saver = tf.train.Saver(max_to_keep=3)
        best_saver = tf.train.Saver(max_to_keep=3)

        init = tf.global_variables_initializer()

        global_step = 0
        best_valid_score = 0

        with tf.Session() as sess:
            sess.run(init)

            for i in range(options['epochs']):
                log.info('Epoch {}'.format(i + 1))
                start = datetime.now()

                for batch_num, batch in enumerate(batch_generator(train, options['batch_size'], options)):
                    global_step += 1
                    _, loss, predictions, correct = model.train(batch, sess)

                    if batch_num != 0 and batch_num % 5 == 0:
                        batched_data_size = len(train) / options['batch_size']
                        log.info('updates[{}]  remaining[{}]'.format(batch_num, str(
                            (datetime.now() - start) / (batch_num + 1) * (batched_data_size - batch_num - 1)).split(
                            '.')[0]))

                        score = Score(predictions, correct)
                        log.info("train F1: {}".format(score.computef1()))

                        if batch_num % 10 == 0:
                            for p, t in zip(predictions[:10], correct[:10]):
                                print('prediction={}\ntruth={}'.format(p, t))
                                print('-' * 30)

                    save_opt = options['save_every_n_iter']
                    if batch_num != 0 and batch_num % save_opt == 0:
                        iter_saver.save(sess, options['iter_check'], global_step=global_step)

                    dev_opt = options['dev_test_every_n_iter']
                    if batch_num != 0 and batch_num % dev_opt == 0:
                        predictions = []
                        for batch in batch_generator(dev, options['batch_size'], options, test_mode=True):
                            predictions.extend(model.test(batch, sess))
                        for p, t in zip(predictions[:10], dev_answers[:10]):
                            print('prediction={}\ntruth={}'.format(p, t))
                            print('-' * 30)

                        score = Score(predictions, dev_answers)
                        f1_score = score.computef1()
                        log.info("train F1: {}".format(f1_score))
                        if f1_score > best_valid_score:
                            best_saver.save(sess, options['best_check'], global_step=global_step)
                            best_valid_score = f1_score

if __name__ == '__main__':
    main()
