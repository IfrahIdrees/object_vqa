
import tensorflow as tf
import tf.keras.layers as layers
from slot_attention.model import SlotAttentionEncoder

class ocvqa(layers.Layer):
    def __init__(self, vocab_size, answer_vocab_size, resolution, slot_num_iterations=3, num_slots=10, relation_dim=256, response_dims=[256, 256, 29]):
        self.objects = SlotAttentionEncoder(resolution, num_iterations=slot_num_iterations, num_slots=num_slots)
        self.question = q_encoder(vocab_size)
        self.relation_layer = layers.Sequential()
        for _ in range(4):
            self.relation_layer.add(layers.Dense(relation_dim, activation='relu'))
        
        self.response_layer = layers.Sequential()
        self.response_layer.add(layers.Dense(response_dims[0], activation='relu'))
        self.response_layer.add(layers.Dense(response_dims[1], activation='relu'))
        self.response_layer.add(layers.Dropout(0.5))
        self.response_layer.add(layers.Dense(response_dims[2], activation='relu'))
        self.linear_layer = layers.Dense(answer_vocab_size)
        self.output = layers.softmax()
        

    def call(self, imgs, q):
        '''
            imgs: batch of input images
            q: batch of questions
        '''
        objects = self.objects(imgs)  # extract objects (batch, slots, slot_dim)
        questions = self.question(q) # encode questions (batch, encoder_dim)
        
        # Cartesian product of objects and question.
        batch_size, n_slots, _ = objects.shape
        object_1 = tf.tile(objects[:, :, None, :], [1, 1, n_slots, 1])
        object_2 = tf.tile(objects[:, None, :, :], [1, n_slots, 1, 1])
        object_pairs = tf.reshape(tf.concat([object_1, object_2], axis=3), [batch_size, n_slots**2, -1]) # (batch, slots*slots, slot_dim*2)
        questions = tf.tile(questions[:, None, :], [1, n_slots**2, 1])
        object_q = tf.concat([object_pairs, questions], axis=2)

        relations = tf.reduce_sum(self.relation(object_q), axis=1) # (batch, relation_dim)
        
        return self.output(self.linear_layer(self.response_layer(relations))) # prediction
        

class q_encoder(layers.Layer):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=128):
        self.embedding = layers.Embedding(vocab_size, embedding_size)
        self.rnn = layers.LSTM(hidden_size)

    def call(self, x):
        return self.rnn(self.embedding(x))