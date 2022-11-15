from tqdm import tqdm 
import random 
import torch
from torch.autograd import Variable
import numpy as np

def make_noisy_data(mode):
    if mode == 'train':
        img_size = 28
        n_fake_img = 1000
        fake_img = []

        for i in tqdm(range(n_fake_img)):
            fake_img.append(np.random.randn(img_size * img_size).reshape(1, img_size, img_size))

        fake_img = torch.FloatTensor(fake_img)
        fake_img = fake_img.view(n_fake_img, img_size * img_size)
        fake_img = Variable(fake_img).cuda()
    else:
        img_size = 28
        n_fake_img = 60000
        fake_img = []

        for i in tqdm(range(n_fake_img)):
            fake_img.append(np.random.randn(img_size * img_size).reshape(1, img_size, img_size))

        fake_img = torch.FloatTensor(fake_img)
        fake_img = fake_img.view(n_fake_img, img_size * img_size)
        fake_img = Variable(fake_img).cuda()

    return fake_img