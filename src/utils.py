import __init__
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf

import slot_attention.data as data_utils
import slot_attention.model as model_utils
import slot_attention.utils as utils

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", "pretrained_weights/slot_attention_encoder",
                    "Path to model checkpoint.")
flags.DEFINE_integer("batch_size", 500, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")


def build_encoder():
    resolution = (128, 128)
    slot_encoder = model_utils.SlotAttentionEncoder(resolution=resolution, num_slots=FLAGS.num_slots, num_iterations=FLAGS.num_iterations)
    input_layer = tf.keras.Input(resolution + (3,), FLAGS.batch_size)
    output_layer = slot_encoder(input_layer)
    encoder_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    encoder_model.load_weights(FLAGS.checkpoint_dir + "/slot_attention_object_discovery_encoder")
    return encoder_model.layers[1]

def main(argv):
    print(build_encoder())

if __name__ == "__main__":
  app.run(main)