#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
python src/train.py --job_number 4 --batch_size 64 --learning_rate 0.0001 --num_train_steps 50000 --freeze_slot False 
python src/train.py --job_number 5 --batch_size 64 --learning_rate 0.001  --num_train_steps 50000 --freeze_slot False 

