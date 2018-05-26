import tensorflow as tf
from layers import *

class DRQA:
    def __init__(self, options, embeddings):
        c_len = options['context_len']
        f_num = options['features_num']
        q_len = options['question_len']
        self.context = tf.placeholder(tf.int32, (None, c_len), name='context_ids')
        self.c_features = tf.placeholder(tf.float32, (None, c_len, f_num), name='context_feat')
        self.c_pos = tf.placeholder(tf.int32, (None, c_len), name='context_pos')
        self.c_ner = tf.placeholder(tf.int32, (None, c_len), name='context_ner')
        self.c_mask = tf.placeholder(tf.bool, (None, c_len), name='context_mask')
        self.question = tf.placeholder(tf.int32, (None, q_len), name='question_ids')
        self.q_mask = tf.placeholder(tf.bool, (None, q_len), name='question_mask')
        self.start_true = tf.placeholder(tf.int32, (None), name='start_answer')
        self.end_true = tf.placeholder(tf.int32, (None), name='end_answer')

        with tf.variable_scope('Underlying_architecture'):
            thresh_tune = options['thresh_tune']
            pos_num = options['pos_num']
            ner_num = options['ner_num']
            pos_dim = options['pos_dim']
            ner_dim = options['ner_dim']

            var_embed = tf.Variable(embeddings[:thresh_tune])
            const_embed = tf.Variable(embeddings[thresh_tune:], trainable=False)
            embeddings = tf.concat((var_embed, const_embed), axis=0, name='word_embeddings')
            pos_emb = tf.Variable(tf.truncated_normal((pos_num, pos_dim)), name='pos_embeddings')
            ner_emb = tf.Variable(tf.truncated_normal((ner_num, ner_dim)), name='ner_embeddings')

            context_emb = tf.nn.embedding_lookup(embeddings, self.context)
            question_emb = tf.nn.embedding_lookup(embeddings, self.question)

            aligned_q_layer = AlignedQuestionEmbedding(context_emb, question_emb)
            aligned_q_emb = aligned_q_layer.aligned_embeddings
            c_pos_emb = tf.nn.embedding_lookup(pos_emb, self.c_pos)
            c_ner_emb = tf.nn.embedding_lookup(ner_emb, self.c_ner)

            input_list = [context_emb, self.c_features, c_pos_emb, c_ner_emb, aligned_q_emb]
            full_context_encoding = tf.concat(input_list, axis=2)

            hidden_units = options['hidden_size']
            layers_num = options['layers_num']
            dropout_rate = options['dropout_rate']

            with tf.variable_scope('Context_encoder'):
                context_rnn = StackedBRNN(full_context_encoding, hidden_units, layers_num, dropout_rate)
                context_rnn_outs = context_rnn.output

            with tf.variable_scope('Question_encoder'):
                question_rnn = StackedBRNN(question_emb, hidden_units, layers_num, dropout_rate)
                question_rnn_outs = question_rnn.output

                q_attention = QuestionAttention(question_rnn_outs, self.q_mask)
                q_attention_enc = q_attention.encodings

            with tf.variable_scope('Decoder'):
                start_biattn_layer = BilinearAttention(context_rnn_outs, q_attention_enc, self.c_mask)
                end_biattn_layer = BilinearAttention(context_rnn_outs, q_attention_enc, self.c_mask)

                self.start_score = start_biattn_layer.score
                self.end_score = end_biattn_layer.score

        with tf.variable_scope('Probabilities'):
            #mask = tf.cast(tf.logical_not(self.c_mask), tf.float32)
            mask = tf.cast(self.c_mask, tf.float32)
            self.start_probs = tf.multiply(tf.nn.softmax(self.start_score), mask)
            self.end_probs = tf.multiply(tf.nn.softmax(self.end_score), mask)

        with tf.name_scope('loss'):
            start_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.start_true,
                                                                                       logits=self.start_score))
            end_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.end_true,
                                                                                     logits=self.end_score))
            self.loss = start_loss + end_loss

        learning_rate = options['learning_rate']
        clipping_value = options['grad_clipping']
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        grad_vars = optimizer.compute_gradients(self.loss)
        clipped_grad_vars = [(tf.clip_by_norm(grad, clipping_value), var) for grad, var in grad_vars]

        self.train_step = optimizer.apply_gradients(clipped_grad_vars)

    def predict(self, batch, start_probs, end_probs, y_s=None,y_e=None):
        def _get_best_span_for_context(start_probs, end_probs):
            max_len = 15
            best_prob = 0.0
            best_span = (None, None)
            for start in range(len(start_probs)):
                for end in range(start, min(len(start_probs), start + max_len)):
                    s_prob = start_probs[start]
                    e_prob = end_probs[end]
                    span_prob = s_prob * e_prob

                    if span_prob > best_prob:
                        best_prob = span_prob
                        best_span = (start, end)
            return best_span

        text = batch[-2]
        spans = batch[-1]
        predictions = []
        y_text = []
        for i in range(len(start_probs)):
            start_idx, end_idx = _get_best_span_for_context(start_probs[i], end_probs[i])
            start_offset, end_offset = spans[i][start_idx][0], spans[i][end_idx][1]
            predictions.append(text[i][start_offset:end_offset])

            if y_s is not None and y_e is not None:
                y_s_off = spans[i][y_s[i]][0]
                y_e_off = spans[i][y_e[i]][1]
                y_text.append([text[i][y_s_off:y_e_off]])

        return predictions, y_text

    def train(self, batch, sess):
        feed_dict = {
            self.context: batch[0],
            self.c_features: batch[1],
            self.c_pos: batch[2],
            self.c_ner: batch[3],
            self.c_mask: batch[4],
            self.question: batch[5],
            self.q_mask: batch[6],
            self.start_true: batch[7],
            self.end_true: batch[8]
        }

        ops = [self.train_step, self.loss, self.start_probs, self.end_probs]
        tr_op, loss, start_probs, end_probs = sess.run(ops, feed_dict=feed_dict)

        predictions, y_true = self.predict(batch, start_probs, end_probs, batch[7], batch[8])
        return tr_op, loss, predictions, y_true

    def test(self, batch, sess):
        feed_dict = {
            self.context: batch[0],
            self.c_features: batch[1],
            self.c_pos: batch[2],
            self.c_ner: batch[3],
            self.c_mask: batch[4],
            self.question: batch[5],
            self.q_mask: batch[6]
        }

        ops = [self.start_probs, self.end_probs]
        start_probs, end_probs = sess.run(ops, feed_dict=feed_dict)
        predictions, _ = self.predict(batch, start_probs, end_probs)
        return predictions



