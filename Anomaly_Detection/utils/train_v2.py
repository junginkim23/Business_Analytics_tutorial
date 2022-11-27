import time 
from collections import defaultdict
from datetime import timedelta
import matplotlib.pyplot as plt
from torch.autograd import Variable
import visdom
import numpy as np
import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import os 

class Trainer:
    def __init__(self,args,loader,model,criterion,opt):
        self.args = args
        self.loader = loader
        self.model = model
        self.criterion = criterion 
        self.optimizer = opt

    def train(self):
        self.model.train()

        for epoch in range(self.args.epochs):
            train_loss = 0.0
            start = time.time()
            for i, data in enumerate(self.loader):
                X = data[0].to(self.args.device)
                y = data[1].to(self.args.device)

                pred = self.model(X)

                loss = self.criterion(pred,X)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss = train_loss/(i+1)
            print(f'epoch [{epoch+1}/{self.args.epochs}] | loss:{float(train_loss):.4f} | Time {time.time()-start:.4f}')


            if (epoch+1)%20==0:
                self._save_model(epoch)

    def _save_model(self,epoch):
        save_dict={
            'epoch': epoch,
            'model': self.model.state_dict()
        }

        save_file = os.path.join(self.args.ckpt_dir, f'epoch{epoch+1}.pt')
        torch.save(save_dict,save_file)