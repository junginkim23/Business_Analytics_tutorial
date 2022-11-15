import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import time
from utils.AE import AE
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchvision import transforms 
from utils.args import get_args
from utils.add_noisy import make_noisy_data
from utils.train import Trainer
from utils.test import Tester
import visdom

def main():

    args = get_args()

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = MNIST('./data',transform=img_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)

    train_fake_img = make_noisy_data('train')
    test_fake_img = make_noisy_data('test')

    model = AE()
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_d)

    trainer = Trainer(args,train_fake_img,dataloader,model,criterion,optimizer)
    trainer.train()
    tester = Tester(args,test_fake_img,dataloader,model)
    tester.test()

if __name__ == '__main__':
    main()