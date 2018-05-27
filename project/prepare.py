import re
import os
import urllib.request
import json
import msgpack
import unicodedata
import numpy as np
import spacy
import pandas as pd
import argparse
import collections
import logging
from tqdm import tqdm

wv_dim = 300

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def normalize_text(text):
    return unicodedata.normalize('NFD', text)


def load_wv_vocab(file):
    vocab = set()
    with open(file, 'r', encoding="utf-8") as f:
        for line in f:
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))  # a token may contain space
            vocab.add(token)
    return vocab


def flatten_json(file, mode):
    def _proc(article):
        rows = []
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if mode == 'train':
                    answer = answers[0]['text']
                    answer_start = answers[0]['answer_start']
                    answer_end = answer_start + len(answer)
                    rows.append((id_, context, question, answer, answer_start, answer_end))
                else:
                    answers = [a['text'] for a in answers]
                    rows.append((id_, context, question, answers))
        return rows

    with open(file) as f:
        data = json.load(f)['data']

    rows = map(_proc, data)
    rows = sum(rows, [])
    return rows


def get_answer_index(context_doc, answer_start, answer_end):
    id_start = id_end = -1
    for i, _ in enumerate(context_doc):
        if context_doc[i].idx == answer_start:
            id_start = i
        if context_doc[i].idx >= answer_end:
            id_end = i - 1
            if id_start != -1:
                return id_start, id_end
            else:
                return None, None
    return None, None


def normalize_spaces(text):
    text = re.sub('\s+', ' ', text)
    return text


def build_vocab(questions, contexts, wv_vocab):
    counter = collections.Counter(w for doc in questions + contexts for w in doc)
    vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)

    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter


def token2id(docs, vocab, unk_id=None):
    w2id = {w: i for i, w in enumerate(vocab)}
    ids = [[w2id[w] if w in w2id else unk_id for w in doc] for doc in docs]
    return ids


def build_embedding(embed_file, targ_vocab, dim_vec):
    vocab_size = len(targ_vocab)
    emb = np.zeros((vocab_size, dim_vec))
    w2id = {w: i for i, w in enumerate(targ_vocab)}
    with open(embed_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]
    return emb


def preprocess(log, args):
    extra_dir = './extra_data/'
    if not os.path.exists(extra_dir):
        os.makedirs(extra_dir, exist_ok=True)

    squad_train = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'
    squad_dev = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'

    log.info('Downloading train dataset')
    urllib.request.urlretrieve(squad_train, extra_dir+'/train.json')
    log.info('Downloading dev dataset')
    urllib.request.urlretrieve(squad_dev, extra_dir + '/dev.json')
    log.info('Successfully downloaded datasets')

    wv_file = args.embeddings
    if wv_file == '':
        log.info('Downloading glove embeddings (~2Gb)')
        glove_840B = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
        urllib.request.urlretrieve(glove_840B, extra_dir + '/glove.zip')
        log.info('Embeddings loaded, unzipping')
        os.system("unzip extra_data/folder.zip -d ./extra_data/")
        log.info('Successfully unzipped')
        wv_file = extra_dir+'glove.840B.300d.txt'

    train_file = extra_dir+'train.json'
    dev_file = extra_dir+'dev.json'

    log.info('Loading vocabulary')
    wv_vocab = load_wv_vocab(wv_file)
    log.info('Vocabulary loaded')

    train = flatten_json(train_file, mode='train')
    train = pd.DataFrame(train, columns=['id', 'context', 'question', 'answer', 'answer_start', 'answer_end'])
    dev = flatten_json(dev_file, mode='dev')
    dev = pd.DataFrame(dev, columns=['id', 'context', 'question', 'answers'])
    log.info('Json data flattened.')

    nlp = spacy.load('en', disable=['parser', 'tagger', 'entity'])
    context_iter = (normalize_spaces(c) for c in train.context)
    log.info('Initial processing started, expect to process {} docs'.format(len(train.context)))
    context_docs = [doc for doc in tqdm(nlp.pipe(
        context_iter))]
    log.info('Got initial docs.')

    train['answer_start_token'], train['answer_end_token'] = zip(*[get_answer_index(a, b, c) for a, b, c in
                                                                   zip(context_docs, train.answer_start,
                                                                       train.answer_end)])
    initial_len = len(train)
    train.dropna(inplace=True)
    log.info('Dropped {} inconsistent samples.'.format(initial_len - len(train)))

    nlp = spacy.load('en')

    questions = list(train.question) + list(dev.question)
    contexts = list(train.context) + list(dev.context)
    log.info('train_size = {}, dev.size = {}'.format(train.question.size, dev.question.size))

    context_text = [normalize_spaces(c) for c in contexts]
    question_text = [normalize_spaces(q) for q in questions]

    log.info('Start processing questions.')
    log.info('Expect to process {} docs'.format(len(train.context) + len(dev.context)))
    question_docs = [doc for doc in tqdm(nlp.pipe(question_text))]
    log.info('Finished processing questions.')
    log.info('Start processing contexts.')
    context_docs = [doc for doc in tqdm(nlp.pipe(context_text))]
    log.info('Finished processing contexts.')

    question_tokens = [[normalize_text(w.text) for w in doc] for doc in question_docs]
    context_tokens = [[normalize_text(w.text) for w in doc] for doc in context_docs]

    log.info('Generating features')
    context_token_span = [[(w.idx, w.idx + len(w.text)) for w in doc] for doc in tqdm(context_docs)]
    context_tags = [[w.tag_ for w in doc] for doc in tqdm(context_docs)]
    context_ents = [[w.ent_type_ for w in doc] for doc in tqdm(context_docs)]
    context_features = []
    for question, context in tqdm(zip(question_docs, context_docs)):
        question_word = {w.text for w in question}
        question_lower = {w.text.lower() for w in question}
        question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}
        match_origin = [w.text in question_word for w in context]
        match_lower = [w.text.lower() in question_lower for w in context]
        match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in context]
        context_features.append(list(zip(match_origin, match_lower, match_lemma)))
    log.info('Generated')

    vocab, counter = build_vocab(question_tokens, context_tokens, wv_vocab)

    question_ids = token2id(question_tokens, vocab, unk_id=1)
    context_ids = token2id(context_tokens, vocab, unk_id=1)

    context_tf = []
    for doc in context_tokens:
        counter_ = collections.Counter(w.lower() for w in doc)
        total = sum(counter_.values())
        context_tf.append([counter_[w.lower()] / total for w in doc])

    context_features = [[list(w) + [tf] for w, tf in zip(doc, tfs)] for doc, tfs in
                        zip(context_features, context_tf)]

    vocab_tag = list(nlp.tagger.labels)
    context_tag_ids = token2id(context_tags, vocab_tag)
    counter_ent = collections.Counter(w for doc in context_ents for w in doc)
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags'.format(len(vocab_ent)))
    context_ent_ids = token2id(context_ents, vocab_ent)

    embedding = build_embedding(wv_file, vocab, wv_dim)
    log.info('Embedding matrix built')

    train.to_csv('./extra_data/train.csv', index=False, encoding="utf-8")
    dev.to_csv('./extra_data/dev.csv', index=False, encoding="utf-8")

    meta = {
        'vocab': vocab,
        'embedding': embedding.tolist(),
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent
    }
    with open('./extra_data/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)

    result = {
        'trn_question_ids': question_ids[:len(train)],
        'dev_question_ids': question_ids[len(train):],
        'trn_context_ids': context_ids[:len(train)],
        'dev_context_ids': context_ids[len(train):],
        'trn_context_features': context_features[:len(train)],
        'dev_context_features': context_features[len(train):],
        'trn_context_tags': context_tag_ids[:len(train)],
        'dev_context_tags': context_tag_ids[len(train):],
        'trn_context_ents': context_ent_ids[:len(train)],
        'dev_context_ents': context_ent_ids[len(train):],
        'trn_context_text': context_text[:len(train)],
        'dev_context_text': context_text[len(train):],
        'trn_context_spans': context_token_span[:len(train)],
        'dev_context_spans': context_token_span[len(train):]
    }

    with open('./extra_data/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)

    log.info('Processing completed')


def download_data(log, args):
    log.info('Downloading preprocessed data')
    extra_data_link = 'https://www.dropbox.com/sh/th0f5ha1vgy7xlf/AAAXkI8wUAcUDxLhikaMlqKua?dl=1'
    urllib.request.urlretrieve(extra_data_link, './extra_data.zip')
    os.system("unzip extra_data.zip -d ./extra_data/")
    os.system("rm extra_data.zip")

    if args.load_embed:
        log.info('Downloading glove embeddings (~2Gb)')
        glove_840B = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
        urllib.request.urlretrieve(glove_840B, './extra_dir/glove.zip')
        log.info('Embeddings loaded, unzipping')
        os.system("unzip extra_data/folder.zip -d ./extra_data/")
        log.info('Successfully unzipped')

    log.info('Extra data loaded')

    log.info('Downloading model weights')
    model_weight_link = 'https://www.dropbox.com/sh/inftfkkhetyqcna/AACmk_qNpbCqncGG_e-njvwna?dl=1'
    urllib.request.urlretrieve(model_weight_link, './model.zip')
    os.system("unzip model.zip -d ./model/")
    os.system("rm model.zip")

def main():
    parser = argparse.ArgumentParser(description='Preprocessing data files')
    parser.add_argument('--embeddings', default='', help='Path to glove embeddings file')
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--load_embed', type='bool', default=False)
    parser.add_argument('--preprocess', type='bool', default=False)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)

    if args.preprocess:
        preprocess(log, args)
        return
    else:
        download_data(log, args)
        return


if __name__ == '__main__':
    main()
