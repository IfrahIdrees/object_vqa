import __init__

import datetime
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

import slot_attention.data as data_utils
import slot_attention.model as model_utils
import slot_attention.utils as utils
import json
# FLAGS = flags.FLAGS
# flags.DEFINE_string("checkpoint_dir", "pretrained_weights/slot_attention_encoder",
#                     "Path to model checkpoint.")
# flags.DEFINE_integer("batch_size", 500, "Batch size for the model.")
# flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
# flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")


def build_object_encoder(num_slots=10, num_iterations=3, checkpoint_dir="pretrained_weights/slot_attention_encoder", batch_size=500):
    resolution = (128, 128)
    slot_encoder = model_utils.SlotAttentionEncoder(resolution=resolution, num_slots=num_slots, num_iterations=num_iterations)
    input_layer = tf.keras.Input(resolution + (3,), batch_size=batch_size)
    output_layer = slot_encoder(input_layer)
    #output_layer.trainable = False
    #output_layer.layers[-1].trainable = True 
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    encoder_model.trainable = False
    encoder_model.layers[-1] = True
    encoder_model.load_weights(checkpoint_dir + "/slot_attention_object_discovery_encoder")
    return encoder_model

def accuracy(targets, probs):
    results = tf.argmax(probs, axis=-1)
    return tf.reduce_mean(tf.cast(tf.math.equal(results, targets), dtype=tf.float64))


def inverse_vocabulary(vocabulary):
    return dict(zip(range(len(vocabulary)), vocabulary))

def sequence_to_text(sequence, inverse_vocabulary):
    return ' '.join(map(lambda idx: inverse_vocabulary[idx], list(sequence)))

def save_vocab(vocabulary, name, directory='./checkpoints'):
    with open(directory + '/'+ name, 'w') as file:
        json.dump(vocabulary, file)

def load_vocab(name, directory='./checkpoints'):
    with open(directory + '/'+ name, 'r') as file:
        vocab = json.load(file)
    return vocab

def main(argv):
    print(build_object_encoder(FLAGS.num_slots, FLAGS.num_iterations, FLAGS.checkpoint_dir, FLAGS.batch_size))

if __name__ == "__main__":
    #app.run(main)
    target = tf.constant([0], dtype=tf.int64)
    probs = tf.constant([0.5,0.1, 0.4], dtype=tf.float64)
    print(accuracy(target, probs))
