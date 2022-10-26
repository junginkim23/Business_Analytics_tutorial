import os 
import argparse


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--kernel', default='linear', type=str, choices=['linear','poly','rbf'])
    parser.add_argument('--model', default='svc', type=str, choices=['svc','svr'])

    args = parser.parse_args()

    return args