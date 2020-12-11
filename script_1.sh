#!/bin/bash


export CUDA_VISIBLE_DEVICES=1
python src/train.py --job_number 7 --batch_size 64 --learning_rate 0.000005 --num_train_steps 50000 --freeze_slot False 
python src/train.py --job_number 8 --batch_size 64 --learning_rate 0.005 --num_train_steps 50000 --freeze_slot False 