import __init__
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.preprocessing as preprocess
import slot_attention.data as slot


def preprocess_text(sentences, tokenizer, max_length=32):
    sentences = map(lambda s: s.decode('utf-8'), list(sentences.numpy()))
    tokenizer.fit_on_texts(sentences) # update vocab
    sequences = tokenizer.texts_to_sequences(sentences) # word2idx
    padded = preprocess.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return tf.convert_to_tensor(padded, dtype=tf.int32)

def preprocess_questions(features, q_tokenizer, a_tokenizer, max_q_length=32):
    question = preprocess_text(features['question_answer']['question'], q_tokenizer, max_length=max_q_length)
    answer = preprocess_text(features['question_answer']['answer'], a_tokenizer, max_length=1)
    return {'question': question, 'answer': answer}

def preprocess_clevr(features, q_tokenizer, a_tokenizer, resolution=(128,128), **kwords):
    image_data = slot.preprocess_clevr(features, resolution, **kwords)
    textual_data = preprocess_questions(features, q_tokenizer, a_tokenizer)
    
    return dict(list(image_data.items()) + list(textual_data.items()))

## Functions taken from slot attention 
def build_clevr(split, resolution=(128, 128), max_vocab_size=1000, shuffle=False, max_n_objects=10,
                num_eval_examples=512, get_properties=True, apply_crop=False):
    """Build CLEVR dataset."""
    if split == "train" or split == "train_eval":
        ds = tfds.load("clevr:3.0.0", split="train", shuffle_files=shuffle)
        #ds = tfds.load("clevr:3.1.0", split="train", shuffle_files=shuffle)
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

    ds = ds.filter(filter_fn)
    q_tokenizer = preprocess.text.Tokenizer(max_vocab_size=max_vocab_size)
    a_tokenizer = preprocess.text.Tokenizer(max_vocab_size=max_vocab_size)

    def _preprocess_fn(x, resolution, max_n_objects=max_n_objects):
        return preprocess_clevr(
            x, resolution, q_tokenizer, a_tokenizer, apply_crop=apply_crop, get_properties=get_properties,
            max_n_objects=max_n_objects)
    ds = ds.map(lambda x: _preprocess_fn(x, resolution))
    return ds, (q_tokenizer, a_tokenizer)

def build_clevr_iterator(batch_size, split, **kwargs):
    ds, tokenizers = build_clevr(split=split, **kwargs)
    ds = ds.repeat(-1)
    ds = ds.batch(batch_size, drop_remainder=True)
    return iter(ds), tokenizers

if __name__=="__main__":
    ds, tokenizers = build_clevr('train[:10]')
    print(tokenizers[0].word_index)
    print(tokenizers[1].word_index)
    for d in ds:
        print(d)
    