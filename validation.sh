# !/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 1 --checkpoint 10021 --cuda
