import __init__

import tensorflow as tf
import tensorflow.keras.layers as layers

from slot_attention.model import SlotAttentionEncoder
from utils import build_object_encoder



class ocvqa(layers.Layer):
    def __init__(self, vocab_size, answer_vocab_size, relation_dim=256, response_dims=[256, 256, 29]):
        super(ocvqa, self).__init__()
        self.question = q_encoder(vocab_size)
        self.relation_layer = tf.keras.Sequential()
        for _ in range(4):
            self.relation_layer.add(layers.Dense(relation_dim, activation='relu'))
        
        self.response_layer = tf.keras.Sequential()
        self.response_layer.add(layers.Dense(response_dims[0], activation='relu'))
        self.response_layer.add(layers.Dense(response_dims[1], activation='relu'))
        self.response_layer.add(layers.Dropout(0.5))
        self.response_layer.add(layers.Dense(response_dims[2], activation='relu'))
        self.linear_layer = layers.Dense(answer_vocab_size)
        self.out = layers.Softmax(axis=1)
        
    def call(self, objects, q):
        '''
            objects : batch of objects (as from slot_attention)
            q: batch of questions
        '''
        questions = self.question(q) # encode questions (batch, encoder_dim)
        
        # Cartesian product of objects and question.
        batch_size, n_slots, _ = objects.shape
        object_1 = tf.tile(objects[:, :, None, :], [1, 1, n_slots, 1])
        object_2 = tf.tile(objects[:, None, :, :], [1, n_slots, 1, 1])
        object_pairs = tf.reshape(tf.concat([object_1, object_2], axis=3), [batch_size, n_slots**2, -1]) # (batch, slots*slots, slot_dim*2)
        questions = tf.tile(questions[:, None, :], [1, n_slots**2, 1])
        object_q = tf.concat([object_pairs, questions], axis=2)
       
        relations = tf.reduce_sum(self.relation_layer(object_q), axis=1) # (batch, relation_dim)
        logits = self.linear_layer(self.response_layer(relations))

        # print(objects)
        # print(object_pairs)
        # print(questions)
        # print(object_q)
        # print(relations)
        # print(logits)

        return self.out(logits) # prediction
        

class q_encoder(layers.Layer):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=128):
        super(q_encoder, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.rnn = layers.LSTM(hidden_size)

    def call(self, q):
        return self.rnn(self.embedding(q))


def build_ocvqa_model(vocab_size, answer_vocab_size, question_max_length=32, resolution = (128, 128), slot_num_iterations=3, num_slots=10, relation_dim=256, response_dims=[256, 256, 29], batch_size=500):
    #images = tf.keras.Input(resolution + (3,), batch_size=batch_size)
    questions = tf.keras.Input(question_max_length, batch_size=batch_size)
    objects = build_object_encoder(batch_size=batch_size)
    objects.trainable = False
    vqahead = ocvqa(vocab_size, answer_vocab_size, relation_dim, response_dims)(objects.layers[1].output, questions)
    return tf.keras.Model(inputs=(objects.input, questions), outputs=vqahead)


if __name__ == "__main__":
    print(build_ocvqa_model(100, 10).summary())
