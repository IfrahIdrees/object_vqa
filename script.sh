#!/bin/bash



python src/train.py --job_number 0 --batch_size 64 --learning_rate 0.00004 --num_train_steps 985 > out_job_0.txt
python src/train.py --job_number 1 --batch_size 64 --learning_rate 0.0004 --num_train_steps 985 > out_job_1.txt
python src/train.py --job_number 2 --batch_size 64 --learning_rate 0.005 --num_train_steps 985 > out_job_2.txt
python src/train.py --job_number 3 --batch_size 128 --learning_rate 0.00004 --num_train_steps 492 > out_job_3.txt
python src/train.py --job_number 4 --batch_size 512 --learning_rate 0.00004 --num_train_steps 123 > out_job_4.txt

