# OCVQA - An Object-Centric Approach to Visual Question Answering
Course Project CS2950K

## Requirements

Make sure to have a GPU-compatible version of TensorFlow (`>= 2.2.0`) installed
and install TensorFlow Datasets (`pip install tfds-nightly`) to load the
[CLEVR dataset](https://cs.stanford.edu/people/jcjohns/clevr/). Lastly, make
sure you have the `absl-py` package installed: `pip install absl-py`.
Alternatively you can run `pip3 install -r requirements.txt` (see `run.sh`).

The code was tested on a single GPU's with 8GB of memory. Consider reducing the
batch size to train the model on GPUs with less memory.

NOTE: Executing the code (training or evaluation) for the first time will
download the full CLEVR dataset (17.7GB).

This is link to the [Slot Attention Module](https://github.com/google-research/google-research/tree/master/slot_attention) codebase. Here is the reference to the codebase for [Relation Network](https://github.com/clvrai/Relation-Network-Tensorflow). We do our own implementation of Relation Network for the VQA downstream task.

## OCVQA Directory Structure
The directory structure to run the training scripts is as follows:
1) Git clone slot attention module.
1) Navigate to the the parent directory (`google-research`)  and clone the folders of our codebase at the same level as slot attention folder. 
2) src folder - contains all the source code for our OCVQA model <br/>
   pretained_weights folder - contains the pretained weights of slot attention module for the task of object discovery. <br/>
   Checkpoints folder - contains all the checpoints during training of OCVQA

## Training OCVQA
To train OCVQA you can run either of (script.sh, script_0.sh or script_1.sh). These files are to be run in the parent directory
(`google-research`). 
Script.sh runs the model sequentially on whichever GPU is available.
To run jobs on GPU-0 and GPU-1 use scripe_0.sh and script_1.sh respectively.
