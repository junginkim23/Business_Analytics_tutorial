import time 
from collections import defaultdict
from datetime import timedelta
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import numpy as np
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import os 

class Tester:
    def __init__(self,args,imgs,loader,model):
        self.args = args
        self.fake_imgs = imgs
        self.loader = loader
        self.model = model
    
    def test(self):
        save_file = os.path.join(self.args.ckpt_dir,f'epoch{self.args.epochs}.pt')
        ckpt = torch.load(save_file)
        self._load_model(ckpt)

        with torch.no_grad():
            self.model.eval()

            # fake img
            pred_ab = self.model(self.fake_imgs)
            fake = (self.fake_imgs-pred_ab).data.cpu().numpy()
            fake = np.sum(fake**2, axis=1)
            print(f'fake img loss 최대값 : {fake.max()}')
            # normal img
            img = self.loader.dataset.data
            img = img.view(img.size(0),-1)
            img = img.type('torch.cuda.FloatTensor')
            img = img / 255

            pred = self.model(img)

            real = (img - pred).data.cpu().numpy()
            real = np.sum(real**2,axis=1)
            print(f'normal img loss 최대값 : {real.max()}')

        self.make_plt(real,fake)

    def _load_model(self, ckpt):
        self.model.load_state_dict(ckpt['model'])

    
    def make_plt(self,real,fake):
        sns.displot([real,fake],color=['blue','red'])
        plt.yticks(list(range(0,30000,5000)),labels=['0','0.05','0.1','0.15','0.2','0.25'],fontsize=10)
        plt.savefig(os.path.join(self.args.save_img_dir,'img.png'))