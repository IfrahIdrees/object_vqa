import __init__
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.preprocessing as preprocess
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import slot_attention.data as slot
from functools import partial

def preprocess_text(sentences, tokenizer, max_length=32):
    return tokenizer(sentences)

def preprocess_questions(features, q_tokenizer, a_tokenizer, max_q_length=32):
    question = preprocess_text(features['question_answer']['question'], q_tokenizer, max_length=max_q_length)
    answer = preprocess_text(features['question_answer']['answer'], a_tokenizer, max_length=1)
    return {'question': question, 'answer': answer}

def preprocess_clevr(features, q_tokenizer, a_tokenizer, resolution=(128,128), **kwords):
    image_data = slot.preprocess_clevr(features, resolution, **kwords)
    textual_data = preprocess_questions(features, q_tokenizer, a_tokenizer)
    
    return dict(list(image_data.items()) + list(textual_data.items()))

def preprocess_pairs_img_q(features):
    img = features['image']
    q = features['question']
    a = features['answer']
    ds = tf.data.Dataset.from_tensor_slices((q,a))
    ds = ds.map(lambda q, a: {'image': img, 'question': q, 'answer': a})
    return ds

## Functions taken from slot attention 
def build_clevr(split, resolution=(128, 128), max_length=32, max_vocab_size=1000, shuffle=False, max_n_objects=10,
                num_eval_examples=512, get_properties=True, apply_crop=False):
    """Build CLEVR dataset."""
    if split == "train" or split == "train_eval":
        #ds = tfds.load("clevr:3.0.0", split="train", shuffle_files=shuffle)
        ds = tfds.load("clevr:3.1.0", split="train", shuffle_files=shuffle)
        if split == "train":
            ds = ds.skip(num_eval_examples)
        elif split == "train_eval":
        # Instead of taking the official validation split, we take a smaller split
        # from the training dataset to monitor AP scores during training.
            ds = ds.take(num_eval_examples)
    else:
        ds = tfds.load("clevr", split=split, shuffle_files=shuffle)

    def filter_fn(example, max_n_objects=max_n_objects):
        """Filter examples based on number of objects.

        The dataset only has feature values for visible/instantiated objects. We can
        exploit this fact to count objects.

        Args:
        example: Dictionary of tensors, decoded from tf.Example message.
        max_n_objects: Integer, maximum number of objects (excl. background) for
            filtering the dataset.

        Returns:
        Predicate for filtering.
        """
        return tf.less_equal(tf.shape(example["objects"]["3d_coords"])[0],
                            tf.constant(max_n_objects, dtype=tf.int32))

    ds = ds.filter(filter_fn)  # filter by number of objects
    q_vectorization = TextVectorization(max_tokens=max_vocab_size, output_mode='int', output_sequence_length=max_length)
    a_vectorization = TextVectorization(max_tokens=max_vocab_size, output_mode='int', output_sequence_length=1)
    
    question_ds = ds.map(lambda x: x['question_answer']['question'])
    answer_ds = ds.map(lambda x: x['question_answer']['answer'])

    # build vocabs
    q_vectorization.adapt(question_ds)
    a_vectorization.adapt(answer_ds)

    def _preprocess_fn(x, resolution, max_n_objects=max_n_objects):
        return preprocess_clevr(x, q_vectorization, a_vectorization, resolution, apply_crop=apply_crop, get_properties=get_properties,max_n_objects=max_n_objects)
   
    resolution = tf.constant(resolution, dtype=tf.int32)
    ds = ds.map(lambda x: _preprocess_fn(x, resolution))
    ds = ds.interleave(lambda x: preprocess_pairs_img_q(x), cycle_length=5, block_length=1)
    return ds, (q_vectorization, a_vectorization)

def build_clevr_iterator(batch_size, split, **kwargs):
    ds, tokenizers = build_clevr(split=split, **kwargs)
    ds = ds.repeat(-1)
    ds = ds.batch(batch_size, drop_remainder=True)
    print("==========CLEVR BUILT===========")
    return iter(ds), tokenizers

if __name__=="__main__":

    ds, tokenizers = build_clevr('train[:1]')
    print(tokenizers[0].get_vocabulary())
    print(tokenizers[1].get_vocabulary())
    for d in ds:
        print(d)
    