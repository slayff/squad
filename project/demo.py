import time
import logging
import argparse
from collections import Counter
import tensorflow as tf
import spacy
import msgpack
from model import DRQA
from train import batch_generator
from prepare import normalize_spaces, normalize_text

wv_dim = 300

def get_data(context, question, nlp, meta, w2id):
    context_text = normalize_spaces(context)
    question_text = normalize_spaces(question)

    context_doc = nlp(context_text)
    question_doc = nlp(question_text)

    question_tokens = [normalize_text(w.text) for w in question_doc]
    context_tokens = [normalize_text(w.text) for w in context_doc]

    context_token_span = [(w.idx, w.idx + len(w.text)) for w in context_doc]
    context_tags = [w.tag_ for w in context_doc]
    context_ents = [w.ent_type_ for w in context_doc]

    question_word = {w.text for w in question_doc}
    question_lower = {w.text.lower() for w in question_doc}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question_doc}
    match_origin = [w.text in question_word for w in context_doc]
    match_lower = [w.text.lower() in question_lower for w in context_doc]
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context_doc]
    context_features = list(zip(match_origin, match_lower, match_lemma))

    unk_id = 1
    question_ids = [w2id.get(w, unk_id) for w in question_tokens]
    context_ids = [w2id.get(w, unk_id) for w in context_tokens]

    counter_ = Counter(w.lower() for w in context_tokens)
    total = sum(counter_.values())
    context_tf = [counter_[w.lower()] / total for w in context_tokens]
    context_features = context_features + [context_tf]

    vocab_tag = meta['vocab_tag']
    vocab_ent = meta['vocab_ent']
    w2id_tag = {w: i for i, w in enumerate(vocab_tag)}
    w2id_ent = {w: i for i, w in enumerate(vocab_ent)}

    context_tag_ids = [w2id_tag[w] for w in context_tags]
    context_ent_ids = [w2id_ent[w] for w in context_ents]

    result = [context_ids, context_features, context_tag_ids, context_ent_ids, question_ids, context_text, context_token_span]

    return result

def main():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--path', default='./best_checkpoints/', help='Path to model weights (tf checkpoint)')
    args = parser.parse_args()

    options = vars(args)

    with open('extra_data/meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')

    vocab = meta['vocab']
    embeddings = meta['embedding']
    nlp = spacy.load('en')
    w2id = {w: i for i, w in enumerate(vocab)}

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

        'context_len': 1000,
        'features_num': 4,
        'question_len': 100
    }
    options_copy = options.copy()
    options_copy.update(extra_options)
    options = options_copy

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info('Logging initialized')

    graph = tf.Graph()
    with graph.as_default():
        log.info('Loading graph')
        model = DRQA(options, embeddings)
        log.info('Graph loaded')

        save_path = tf.train.latest_checkpoint(options['path'])
        saver = tf.train.Saver()

        with tf.Session() as sess:

            saver.restore(sess, save_path)

            while True:
                try:
                    while True:
                        context = input('Context: ')
                        if context.strip():
                            break
                    while True:
                        question = input('Question: ')
                        if question.strip():
                            break

                    start_time = time.time()
                    data = [get_data(context, question, nlp, meta, w2id)]
                    predictions = []
                    for batch in batch_generator(data, len(data), options, test_mode=True):
                        predictions.extend(model.test(batch, sess))

                    end_time = time.time()
                    print('Answer: {}'.format(predictions[0]))
                    print('Time: {:.4f}s'.format(end_time - start_time))

                except EOFError:
                    print()
                    break

if __name__ == '__main__':
    main()