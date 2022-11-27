import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import time
from utils.AE_v2 import AnomalyDetector
from torch.utils.data import DataLoader
from torchvision import transforms 
from utils.args import get_args
from utils.add_noisy import make_noisy_data
from utils.train_v2 import Trainer
from utils.test_v2 import Tester
import visdom
from torch.utils.data import DataLoader
from utils.dataset import MakeDataset

def main():

    args = get_args()

    train_dataset = MakeDataset(split='train')
    test_dataset = MakeDataset(split='test')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    model = AnomalyDetector(args).to(args.device)
    criterion = nn.MSELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_d)

    trainer = Trainer(args,train_loader,model,criterion,optimizer)
    trainer.train()
    
    tester = Tester(args,test_loader,model)
    pred_df = tester.test()

if __name__ == '__main__':
    main()