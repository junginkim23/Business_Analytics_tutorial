import torch 
import torch.nn as nn
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset

class MakeDataset(Dataset):

    def __init__(self,split='train'):

        data_all = pd.read_csv('./Anomaly_Detection/data/creditcard_data.csv')

        # split dataset

        data_all['normAmount'] = StandardScaler().fit_transform(data_all['Amount'].values.reshape(-1,1))
        data_all.drop(['Time','Amount'],axis=1,inplace=True)

        y = data_all.loc[:,'Class']
        X = data_all.drop(labels='Class',axis=1)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=77)

        train_labels = train_y.astype(bool)
        # test_labels = test_y.astype(bool)

        self.train_data = torch.tensor(data_all.loc[train_X[~train_labels].index].values).float()
        self.test_data = torch.tensor(data_all.loc[test_X.index].values).float()

        if split == 'train' :
            self.data = self.train_data
        else :
            self.data = self.test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx][:-1], self.data[idx][-2].to(torch.int32)

       

