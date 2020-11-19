import __init__
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.preprocessing as preprocess
import slot_attention.data as slot


def preprocess_text(sentences, max_length=32, max_vocab_size=1000, unk_token='<unk>'):
    q_tokenizer = preprocess.text.Tokenizer(num_words=max_vocab_size, oov_token=unk_token)
    q_tokenizer.fit_on_texts(sentences) # update vocab
    sequences = q_tokenizer.texts_to_sequences(sentences) # word2idx
    padded = preprocess.sequence.pad_sequence(sequences, maxlen=max_length, padding='post', truncating='post')
    return q_tokenizer.word_index, padded

def preprocess_questions(features, max_q_length=32, max_vocab_size=1000, unk_token='<unk>'):
    q_vocab, questions = preprocess_text(features['question_answer']['question'], max_length=max_q_length, max_vocab_size=max_vocab_size, unk_token=unk_token)
    a_vocab, answers = preprocess_text(features['question_answer']['answer'], max_length=1, max_vocab_size=max_vocab_size, unk_token=unk_token)
    textual_data = {"q_vocab": q_vocab, "questions": questions, "a_vocab": a_vocab, "answers": answers}
    return textual_data

def preprocess_clevr(features, resolution=(128,128), **kwords):
    image_data = slot.preprocess_clevr(features, resolution, **kwords)
    textual_data = preprocess_questions(features)
    
    return image_data, textual_data

## Functions taken from slot attention 

def build_clevr(split, resolution=(128, 128), shuffle=False, max_n_objects=10,
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

    def _preprocess_fn(x, resolution, max_n_objects=max_n_objects):
        return preprocess_clevr(
            x, resolution, apply_crop=apply_crop, get_properties=get_properties,
            max_n_objects=max_n_objects)
    ds = ds.map(lambda x: _preprocess_fn(x, resolution))
    return ds

def build_clevr_iterator(batch_size, split, **kwargs):
    ds = build_clevr(split=split, **kwargs)
    ds = ds.repeat(-1)
    ds = ds.batch(batch_size, drop_remainder=True)
    return iter(ds)

if __name__=="__main__":
    ds = tfds.load("clevr:3.1.0", split='test')
    data, vocabs = preprocess_questions(ds)