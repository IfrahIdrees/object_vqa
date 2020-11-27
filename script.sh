#!/bin/bash


nohup python src/train.py --job_number 0 --batch_size 64 --learning_rate 0.00004 >/dev/null 2>&1 &
wait
nohup python src/train.py --job_number 1 --batch_size 64 --learning_rate 0.0004 >/dev/null 2>&1 &
wait
nohup python src/train.py --job_number 2 --batch_size 64 --learning_rate 0.005 >/dev/null 2>&1 &
wait
nohup python src/train.py --job_number 3 --batch_size 128 --learning_rate 0.00004 >/dev/null 2>&1 &
wait
nohup python src/train.py --job_number 4 --batch_size 512 --learning_rate 0.00004 >/dev/null 2>&1 &


