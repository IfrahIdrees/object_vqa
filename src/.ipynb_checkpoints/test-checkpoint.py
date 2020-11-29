import __init__
import time, datetime
import tensorflow as tf
import ocvqa
import dataset as data_util
import utils
import train

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS
# flags.DEFINE_string("model_dir", "./checkpoints/",
#                     "Where to save the checkpoints.")
# flags.DEFINE_integer("seed", 0, "Random seed.")
# flags.DEFINE_integer("batch_size", 4, "Batch size for the model.")
# flags.DEFINE_integer("num_slots", 7, "Number of slots in Slot Attention.")
# flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
# flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
# flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
# flags.DEFINE_integer("warmup_steps", 10000,
#                      "Number of warmup steps for the learning rate.")
# flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
# flags.DEFINE_integer("decay_steps", 100000,
#                      "Number of steps for the learning rate decay.")
# flags.DEFINE_bool("full_eval", False,
#                   "If True, use full evaluation set, otherwise a single batch.")
# flags.DEFINE_string("job_number", "0", "number of the job runnning")



def main(argv):
    del argv
    # Hyperparameters of the model.
    batch_size = FLAGS.batch_size
    num_slots = FLAGS.num_slots
    num_iterations = FLAGS.num_iterations
    base_learning_rate = FLAGS.learning_rate
    num_train_steps = FLAGS.num_train_steps
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps
    job_number =  FLAGS.job_number
    tf.random.set_seed(FLAGS.seed)
    resolution = (128, 128)

    # load dataset. batch must be a dictionary with keys {"images", "question", "answer"}

    data_iterator, tokenizers = data_util.build_clevr_iterator(batch_size=batch_size, split='train[:30%]')
    test_data_iterator, _ = data_util.build_clevr_iterator(batch_size=batch_size, split='validation', tokenizer=tokenizers)
    vocab_size = len(tokenizers[0].get_vocabulary())
    answer_vocab_size = len(tokenizers[1].get_vocabulary())
    logging.info("vocabulary", tokenizers[1].get_vocabulary())

    optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

    model = ocvqa.build_ocvqa_model(vocab_size, answer_vocab_size, batch_size=batch_size)
    print(model.summary())
    
    # Prepare checkpoint manager.
    checkpoint_dir = FLAGS.model_dir + job_number + "/"
    global_step = tf.Variable(
        0, trainable=False, name="global_step", dtype=tf.int64)
    ckpt = tf.train.Checkpoint(
        network=model, optimizer=optimizer, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=checkpoint_dir, max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    
    if ckpt_manager.latest_checkpoint:
        logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
    else:
        logging.info("Initializing from scratch.")
    loss , accuracy = train.evaluate(data_iterator, model)
    logging.info("Train Loss:", loss, "accuracy", accuracy)
    loss , accuracy = train.evaluate(test_data_iterator, model)
    logging.info("Validation Loss:", loss, "accuracy", accuracy)
    
if __name__ == "__main__":
    app.run(main)