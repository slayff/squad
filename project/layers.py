import tensorflow as tf

class StackedBRNN:
    def __init__(self, inputs, hidden_size, layers_num, dropout_rate):
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        outputs = []
        last_output = inputs
        last_state_fw = None
        last_state_bw = None

        def _cell():
            underlying = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size, use_peepholes=True)
            return tf.nn.rnn_cell.DropoutWrapper(cell=underlying, output_keep_prob=self.dropout_rate)

        for i in range(layers_num):
            with tf.variable_scope('Layer_'+str(i)):
                cell_fw = _cell()
                cell_bw = _cell()

                output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                cell_bw,
                                                                last_output,
                                                                initial_state_fw=last_state_fw,
                                                                initial_state_bw=last_state_bw,
                                                                dtype=tf.float32)
                last_output = tf.concat(output, 2)
                last_state_fw = state[0]
                last_state_bw = state[1]

            outputs.extend(output)

            self.concatenated_outputs = tf.concat(outputs, 2)

        self._output = tf.nn.dropout(self.concatenated_outputs, keep_prob=self.dropout_rate)

    @property
    def output(self):
        return self._output

class BilinearAttention:
    def __init__(self, x, y, x_mask):
        self.x_mask = x_mask

        x_size = x.get_shape().as_list()[2]
        y_size = y.get_shape().as_list()[1]

        with tf.variable_scope('Bilinear_Attention'):
            W = tf.Variable(tf.truncated_normal((y_size, x_size), stddev=0.5))
            b = tf.Variable(tf.truncated_normal([x_size], stddev=1.0))

            yW = tf.add(tf.matmul(y, W), b)
            yW_asmatrix = tf.expand_dims(yW, 2)

            xyW = tf.matmul(x, yW_asmatrix)

            self.xyW_vectorized = tf.squeeze(xyW, 2)
            self.mask = tf.cast(tf.logical_not(x_mask), tf.float32)

            self._score = tf.multiply(self.xyW_vectorized, self.mask)

    @property
    def score(self):
        return self._score


class QuestionAttention:
    def __init__(self, x, x_mask):

        x_size = x.get_shape().as_list()
        item_len = x_size[1]
        enc_dim = x_size[2]

        with tf.variable_scope('Question_Attention'):
            W = tf.Variable(tf.truncated_normal((enc_dim, 1), stddev=0.5))

            flattened_x = tf.reshape(x, (-1, enc_dim))
            intermediate_score = tf.reshape(tf.matmul(flattened_x, W), (-1, item_len))

            mask = tf.cast(tf.logical_not(x_mask), tf.float32)

            intermediate_score = tf.multiply(intermediate_score, mask)

            word_importance = tf.nn.softmax(intermediate_score)

            self.word_importance = tf.expand_dims(word_importance, axis=1)
            self._encodings = tf.squeeze(tf.matmul(self.word_importance, x), axis=1)

    @property
    def encodings(self):
        return self._encodings

class AlignedQuestionEmbedding:
    def __init__(self, x, y):

        x_size = x.get_shape().as_list()
        y_size = y.get_shape().as_list()
        doc_len = x_size[1]
        quest_len = y_size[1]
        dim = x_size[2]

        with tf.variable_scope('Aligned_Question_Embedding'):
            W = tf.Variable(tf.truncated_normal((dim, dim), stddev=0.5))
            b = tf.Variable(tf.truncated_normal([dim], stddev=1.0))

            with tf.variable_scope('Paragraph_mapping'):
                flattened_x = tf.reshape(x, (-1, dim))
                x_mapped = tf.nn.relu_layer(flattened_x, W, b)
                x_mapped = tf.reshape(x_mapped, (-1, doc_len, dim))

            with tf.variable_scope('Question_mapping'):
                flattened_y = tf.reshape(y, (-1, dim))
                y_mapped = tf.nn.relu_layer(flattened_y, W, b)
                y_mapped = tf.reshape(y_mapped, (-1, quest_len, dim))

            with tf.variable_scope('Attention_scores'):
                raw_scores = tf.matmul(x_mapped, y_mapped, transpose_b=True)
                raw_scores = tf.reshape(raw_scores, (-1, quest_len))
                scores = tf.nn.softmax(raw_scores)

                scores = tf.reshape(scores, (-1, doc_len, quest_len))

            self._aligned_embeddings = tf.matmul(scores, y)

    @property
    def aligned_embeddings(self):
        return self._aligned_embeddings
