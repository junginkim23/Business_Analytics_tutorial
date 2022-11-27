import torch.nn as nn

class AnomalyDetector(nn.Module):
    def __init__(self,args):
        super(AnomalyDetector, self).__init__()

        self.args =args 

        self.encoder = nn.Sequential(
        nn.Linear(self.args.input_dim,16),
        nn.ReLU(),
        nn.Linear(16,8),
        nn.ReLU(),
        nn.Linear(8,4),
        nn.ReLU())
        
        self.decoder = nn.Sequential(
        nn.Linear(4,8),
        nn.ReLU(),
        nn.Linear(8,16),
        nn.ReLU(),
        nn.Linear(16,self.args.input_dim),
        nn.Sigmoid())


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded