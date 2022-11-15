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
    def __init__(self,args,imgs,loader,model,criterion,opt):
        self.args = args
        self.loader = loader
        self.model = model
        self.fake_imgs = imgs
        self.criterion = criterion 
        self.optimizer = opt
        self.vis = visdom.Visdom()

        self.normal = self.vis.line(X=torch.zeros((1,)).cpu(),
                                    Y=torch.zeros((1)).cpu(),
                                    opts=dict(xlabel='epoch',
                                              ylabel='Loss',
                                              title='Normal data',
                                              legend=['Loss']))

        self.abnormal = self.vis.line(X=torch.zeros((1,)).cpu(),
                                    Y=torch.zeros((1)).cpu(),
                                    opts=dict(xlabel='epoch',
                                              ylabel='Loss',
                                              title='Abnormal data',
                                              legend=['Loss']))

    def train(self):
        self.model.train()

        for epoch in range(self.args.epochs):
            start = time.time()
            for data in self.loader:
                img, _ = data
                img = img.view(img.size(0), -1)
                img = Variable(img).cuda()

                pred = self.model(img)

                loss = self.criterion(pred,img)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            print(f'epoch [{epoch+1}/{self.args.epochs}] | loss:{float(loss.data):.4f} | Time {time.time()-start:.4f}')
            self.vis.line(Y=[loss.data.cpu().numpy()], X=np.array([epoch]),win=self.normal,update='append')

            if epoch % 10 ==0 :
                pic = self.to_img(pred.cpu().data)
                save_image(pic, os.path.join(self.args.save_img_dir,f'./real_image_{epoch}.png'))

            pred_ab = self.model(self.fake_imgs)
            
            loss = self.criterion(pred_ab, self.fake_imgs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f'fake epoch [{epoch+1}/{self.args.epochs}] | loss:{float(loss.data):.4f} | Time {time.time()-start:.4f}')
            self.vis.line(Y=[loss.data.cpu().numpy()], X=np.array([epoch]), win=self.abnormal, update='append')

            if epoch % 10 == 0:
                pic = self.to_img(pred_ab.cpu().data)
                save_image(pic, os.path.join(self.args.save_img_dir,f'./fake_image_{epoch}.png'))
            
            if (epoch+1)%20==0:
                self._save_model(epoch)

    def to_img(self,x):
        x = 0.5 * (x + 1)
        x = x.clamp(0, 1)
        x = x.view(x.size(0), 1, 28, 28)
        return x

    def _save_model(self,epoch):
        save_dict={
            'epoch': epoch,
            'model': self.model.state_dict()
        }

        save_file = os.path.join(self.args.ckpt_dir, f'epoch{epoch+1}.pt')
        torch.save(save_dict,save_file)