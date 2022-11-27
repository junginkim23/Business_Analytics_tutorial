import os 
import argparse
import torch

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--w_d', default=1e-5, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--ckpt_dir', default='./Anomaly_Detection/save_model2', type=str, help='checkpoint directory')
    parser.add_argument('--save_img_dir', default='./Anomaly_Detection/save_img', type=str)
    parser.add_argument('--input_dim', default=29, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)

    args = parser.parse_args()
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return args