# Anomaly Detection

1. Density-based Anomaly Detection
2. Distance-based Anomaly Detection
3. **Model-based Anomaly Detection** (●)

---

This time, we are going to proceed with a tutorial on anomaly detection using autoencoder.

Before diving into the tutorial, what is an autoencoder?

- An autoencoder is an artificial neural network that compresses the input data as much as possible when an input is received and then restores the compressed data back to the original input form. 
- The part that compresses the data is called an encoder, and the part that restores the data is called a decoder. The meaningful data z extracted during the compression process is called a latent vector.

<p align='center'><img src="./save_img/AE.jpg" width='1000' height='200'></p>

---

**python tutorial**

- hypothesis : The autoencoder will have a very low MSE value when normal data comes in. On the other hand, when abnormal data comes in, a high MSE value is derived.

- Since learning was conducted to reduce the difference between the input and output of normal data, of course, if it was well learned, the MSE value according to the input of normal data would get a very low value.

- MNIST data sets is used, and fake images were created for train and inference . MSE Loss was used as the loss function and Adam was used as the optimizer to perform training. As a final result, check the loss plot between the normal image and the abnormal image. In addition, it can be seen that abnormality is detected by checking the distribution of normal and abnormal data.

```
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
```

- [AutoEncoder](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Anomaly_Detection/utils/AE.py)  
    - Six linear layers are stacked on each of the encoder and decoder, and the relu function is used in the encoder and the tanh activation function is used in the decoder.

```
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.enc = nn.Sequential(
            nn.BatchNorm1d(num_features=28*28),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 784),
            nn.BatchNorm1d(num_features=28*28),
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.enc(x)
        decode = self.dec(encode)
        return decode
```

- [train](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Anomaly_Detection/utils/train.py) 

    - Learning is performed with 60,000 normal data, and additional learning is performed with data mixed with some noise.

    - It trains for a total of 160 epochs and stores the weights of the trained model every 20 epochs. In addition, by recording the loss every epoch while learning is in progress, you can check the loss after learning is finished. As can be seen from the results, in the case of fake images, the loss is maintained without being reduced, but in the case of normal images, it is confirmed that the loss is reduced. Finally, a model that has run all 160 epochs is used.

```
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

## output 
epoch [1/160] | loss:0.0668 | Time 13.9934
fake epoch [1/160] | loss:1.0412 | Time 14.0524
epoch [2/160] | loss:0.0597 | Time 12.5840
fake epoch [2/160] | loss:1.0402 | Time 12.6000
epoch [3/160] | loss:0.0520 | Time 12.7024
fake epoch [3/160] | loss:1.0407 | Time 12.7204
epoch [4/160] | loss:0.0467 | Time 12.6946
fake epoch [4/160] | loss:1.0529 | Time 12.7116
epoch [5/160] | loss:0.0416 | Time 12.7174
fake epoch [5/160] | loss:1.0530 | Time 12.7404
epoch [6/160] | loss:0.0408 | Time 12.5694
fake epoch [6/160] | loss:1.0539 | Time 12.6254
epoch [7/160] | loss:0.0386 | Time 12.6462
fake epoch [7/160] | loss:1.0569 | Time 12.6672
epoch [8/160] | loss:0.0406 | Time 12.5580
fake epoch [8/160] | loss:1.0585 | Time 12.5770
```

<p align='center'><img src="./save_img/result.jpg" width='1000' height='300'></p>

- Results of normal and fake images reconstructed using a model trained for 10 epochs and 160 epoch

- epoch 10 

<p align='center'><img src="./save_img/epoch10.jpg" width='1000' height='300'></p>

- epoch 150
<p align='center'><img src="./save_img/epoch150.jpg" width='1000' height='300'></p>

- [inference](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Anomaly_Detection/utils/test.py) 

    - For inference, we use the model trained for 160 epochs and check the MSE displot of normal and fake images. The figure below confirms that the two data are well separated.
    - In the figure, 0 corresponds to a normal image and 1 corresponds to a fake image.
```
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

## output
fake img loss 최대값 : 1.2667418718338013
normal img loss 최대값 : 0.2981035113334656
```
<p align='center'><img src="./save_img/img.png" width='500' height='400'></p>

Conclusion 

- Using MNIST data as normal and randomly creating fake img as abnormal, the distribution difference between normal and abnormal is clearly visible. However, if there is something to be desired, I wanted to evaluate the performance of the model itself with an appropriate threshold by obtaining an anomaly score with the value obtained by calculating the MSE loss function between the input image and the restored image, but It was difficult to try because the distribution of MSE loss values ​​for normal and abnormal data was too different. Although I couldn't do it in this tutorial, I plan to try it before the end of this semester.

---

- 추가 진행사항 

    - 개요 
        - Credit Card Fraud Detection dataset을 사용해서 autoencoder를 사용해 이상 탐지를 진행한다.
        - 기존에 자체적으로 noise를 만들었던 것과 달리 이상(사기) & 정상(사기 x) 레이블에 따른 이상 탐지 진행 
        - 먼저 정상 데이터만으로 학습을 진행했고 추후 이상 데이터와 정상 데이터를 섞은 데이터를 통해 검증을 진행하면서 instance마다 MSELoss를 계산해 anomaly score를 산정한다.
        - threshold는 사용자가 임의로 값을 설정해 해당 값을 넘어가면 이상(1) 그렇지 않으면 (0) 으로 예측하여 최종적으로 f1_score를 통해 평가를 진행한다. 
    
    ```
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
    ```

    - 데이터
        - 데이터의 총수는 284,806개이고 30개의 독립변수와 1개의 종속변수로 존재
        - 30개의 독립변수 중 Time feature는 drop하고 Amount는 Standard Scaler를 사용해 평균이 0이고 분산이 1인 분포를 따르도록 scaling을 진행하여 새로운 변수 normAmount 변수를 추가한다. 그리고 기존에 Amount 변수 또한 drop한다. 
        - 학습과 검증에는 8:2의 비율로 쪼개었고 학습 데이터 중에서 정상만을 사용해 학습을 진행한 후 검증에서는 정상과 이상을 모두 사용하여 검증을 진행하였다. 
    
        ```
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
        ```

    - 모델 학습 
        - 모델은 autoencoder 모델을 사용했고 모델의 전체 구조는 아래 코드에서 확인할 수 있다.
        - epoch 수 또한 하이퍼 파라미터로 설정하여, 실제 학습에는 100 epoch만큼 진행하였다.

        ```
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
        ```

        - epoch 마다 loss와 1 epoch당 걸리는 시간을 출력하였다. 
        ```
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

        ### output ###
        epoch [1/100] | loss:0.9084 | Time 28.9553
        epoch [2/100] | loss:0.8487 | Time 27.1374
        epoch [3/100] | loss:0.8376 | Time 27.1088
        epoch [4/100] | loss:0.8327 | Time 27.2086
        epoch [5/100] | loss:0.8316 | Time 27.3007
        epoch [6/100] | loss:0.8287 | Time 26.9340
        epoch [7/100] | loss:0.8293 | Time 26.8045
        epoch [8/100] | loss:0.8263 | Time 27.2709
        epoch [9/100] | loss:0.8251 | Time 27.2491
                            . 
                            .
                            .
                            .
        epoch [93/100] | loss:0.7860 | Time 29.0389
        epoch [94/100] | loss:0.7863 | Time 27.7789
        epoch [95/100] | loss:0.7861 | Time 27.1986
        epoch [96/100] | loss:0.7866 | Time 29.6684
        epoch [97/100] | loss:0.7903 | Time 27.3582
        epoch [98/100] | loss:0.7865 | Time 27.0228
        epoch [99/100] | loss:0.7920 | Time 27.0310
        epoch [100/100] | loss:0.7891 | Time 27.4149
        ```
    
    - 모델 검증 
        - 최종적으로 100 epoch만큼 학습이 완료된 모델을 불러와 검증 데이터 셋을 사용해 각 instance별 anomaly score를 구하고 임의로 threshold를 선정해 해당 instance가 이상인지 정상인지 예측을 진행하였다. 
        - 평가 지표로는 f1 score를 사용하였다.

        ```
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
        
        ### output ###
        f1 score : 0.007763462173769409
        ```

    - 결론
        - 지난 tutorial에서 아쉬웠던 부분을 직접 python 코드로 작성해보았다. 
        - 처음에 독립 변수 중 Amount 변수의 값 중 가장 큰 값이 348이었다. 해당 변수를 그대로 사용하였을 때 MSELoss 값이 너무 큰 값이 나와 StandardScaler를 사용해야겠다고 생각했다.
        - 사용한 후 loss 값이 1 이하의 값으로 산출된 것을 보고 데이터의 특성을 파악하는 사소한 부분 또한 조심해야 함을 알 수 있었다. 
        - 해당 과정을 진행하면서 새로 생긴 궁금증은 epoch이 지나갈 때마다 생각보다 loss가 많이 줄지 않는다는 것이다. 다음에는 해당 부분에 대해서 고민을 한 번 해봐야 할 것 같다.
        - 아쉬웠던 부분을 해결하고 직접 코드를 짜볼 수 있는 뜻깊은 시간이 된 것 같다.