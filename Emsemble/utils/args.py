import os
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trials', default=20, type = int)
    parser.add_argument('--mode_1', default='vif', type = str)
    parser.add_argument('--mode_2', default='lgb', type = str)
    parser.add_argument('--data_path', default='./data/train.csv', type = str) 

    args = parser.parse_args()

    return args