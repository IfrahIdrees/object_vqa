import __init__
import time, datetime
import tensorflow as tf
import ocvqa
import dataset as data_util
import utils

from absl import app
from absl import flags
from absl import logging
#import os.path
from os import path

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "./checkpoints/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 4, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 7, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 10000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 20000,
                     "Number of steps for the learning rate decay.")
flags.DEFINE_bool("full_eval", False,
                  "If True, use full evaluation set, otherwise a single batch.")
flags.DEFINE_string("job_number", "0", "number of the job runnning")
flags.DEFINE_string("vocab_dir", "./vocab/","Where to save the vocab")
flags.DEFINE_bool("freeze_slot", True,
                  "If True, slot attention module are not tuned.")
@tf.function
def train_step(batch, model, optimizer):
    """Perform a single training step."""
    # print(batch)
    # Get the prediction of the models and compute the loss.
    with tf.GradientTape() as tape:
        preds = model([batch["image"], batch["question"]])
        loss_value = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(batch["answer"], preds, from_logits=False))
        accuracy = utils.accuracy(batch['answer'], preds)
    # Get and apply gradients.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss_value, accuracy

def evaluate(data_iterator, model):
    """Run evaluation."""
    print("In evaluate function")
    if FLAGS.full_eval:  # Evaluate on the full validation set
        test_set_size = 15000
        num_eval_batches = test_set_size // FLAGS.batch_size
    else:
        # By default, we only test on a single batch for faster evaluation.
        test_set_size = FLAGS.batch_size
        num_eval_batches = 1
    loss_value = 0
    accuracy = 0
    
    for _ in tf.range(num_eval_batches):
        batch = next(data_iterator)
        print("questions", batch["question_text"], "answers", batch["answer_text"])
        preds = model([batch["image"], batch["question"]])
        loss_value += tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(batch["answer"], preds, from_logits=False))
        accuracy += FLAGS.batch_size * utils.accuracy(batch['answer'], preds)
    return loss_value/test_set_size, accuracy/test_set_size

def init_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
            print(e)


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
    vocab_dir = FLAGS.vocab_dir
    warmup_steps = FLAGS.warmup_steps
    freeze_slot = FLAGS.freeze_slot
    
    
#     tf.random.set_seed(FLAGS.seed)
    
    resolution = (128, 128)    
    
    # initialize gpus to limit memory growth
    init_gpus()
    
    tokenizers = None
    checkpoint_dir = FLAGS.model_dir + job_number + "/"
    
    question_vocab_file_path = vocab_dir + "question_vocab"
    if path.exists(question_vocab_file_path):
        tokenizers = data_util.load_tokenizers(vocab_dir)
        
    
    # load dataset. batch must be a dictionary with keys {"images", "question", "answer", "question_text", "answer_text"}
    data_iterator, tokenizers = data_util.build_clevr_iterator(batch_size=batch_size, split='train[:100%]', tokenizer=tokenizers)
    test_data_iterator, _ = data_util.build_clevr_iterator(batch_size=batch_size, split='validation', tokenizer=tokenizers)
    vocab_size = len(tokenizers[0].get_vocabulary())
    answer_vocab_size = len(tokenizers[1].get_vocabulary())
    
    if not path.exists(question_vocab_file_path):
        utils.save_vocab(tokenizers[0].get_vocabulary(), 'question_vocab', directory=vocab_dir)
        utils.save_vocab(tokenizers[1].get_vocabulary(), 'answer_vocab', directory=vocab_dir)



#     optimizer = tf.keras.optimizers.Adam(base_learning_rate)
    optimizer = tf.keras.optimizers.SGD(base_learning_rate)
    #optimizer = tf.keras.optimizers.Adam(base_learning_rate)
    model = ocvqa.build_ocvqa_model(vocab_size, answer_vocab_size, batch_size=batch_size, freeze_slot=freeze_slot)
    print(model.summary())
    
    
    
    # Prepare checkpoint manager.
    
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
    
 
    # prepare tensorboard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + job_number+"/"+current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + job_number+"/"+current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    start = time.time()
    for _ in range(num_train_steps):
        batch = next(data_iterator)
        if global_step < warmup_steps:
            learning_rate = base_learning_rate #* tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
        else:
            learning_rate = base_learning_rate
            #learning_rate = learning_rate * (1 / (1 + decay_rate * iteration))
#             learning_rate = learning_rate * (decay_rate ** (tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
        optimizer.lr = base_learning_rate 

        loss_value, accuracy = train_step(batch, model, optimizer)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value, step=global_step)
            tf.summary.scalar('accuracy', accuracy, step=global_step)
        
        if not global_step % 100: # test progress
            loss_value, accuracy = evaluate(test_data_iterator, model)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=global_step)
                tf.summary.scalar('accuracy', accuracy, step=global_step)
            logging.info("Step: %s, Validation Loss: %.6f,  Accuracy: %.4f, Time: %s",
                    global_step.numpy(), loss_value,  accuracy,
                    datetime.timedelta(seconds=time.time() - start))

        # Update the global step. We update it before logging the loss and saving
        # the model so that the last checkpoint is saved at the last iteration.
        global_step.assign_add(1)
        
        # Log the training loss.
        if not global_step % 100:
            logging.info("Step: %s, Loss: %.6f, Accuracy: %.4f,Time: %s",
                    global_step.numpy(), loss_value, accuracy,
                    datetime.timedelta(seconds=time.time() - start))

        # We save the checkpoints every 1000 iterations.
        if not global_step  % 1000:
            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            logging.info("Saved checkpoint: %s", saved_ckpt)


if __name__ == "__main__":
    app.run(main)
