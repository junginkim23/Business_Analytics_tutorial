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
import torch.nn as nn 
import pandas as pd 
from sklearn.metrics import f1_score

class Tester:
    def __init__(self,args,loader,model):
        self.args = args
        self.loader = loader
        self.model = model
        self.loss = nn.MSELoss(reduction='none').to(self.args.device)

    def test(self):
        save_file = os.path.join(self.args.ckpt_dir,f'epoch{self.args.epochs}.pt')
        ckpt = torch.load(save_file)
        self._load_model(ckpt)

        anomaly_scores = []
        targets = [] 

        with torch.no_grad():
            self.model.eval()

            for data in self.loader:
                X = data[0].to(self.args.device)
                targets += data[1].detach().tolist()
    
                pred = self.model(X)
                loss = self.loss(pred,X)
                anomaly_scores += torch.mean(loss,dim=1).detach().cpu().tolist()

        self.df = pd.DataFrame({'instance id':range(1,len(targets)+1),
                                'anomaly scores' : anomaly_scores,
                                'target' : targets})
        self.df['pred'] = np.where(self.df['anomaly scores'] > self.args.threshold, 1, 0)

        f1 = f1_score(self.df['target'],self.df['pred'])
        print(f'f1 score : {f1}')

        return self.df

    def _load_model(self, ckpt):
        self.model.load_state_dict(ckpt['model'])

    
