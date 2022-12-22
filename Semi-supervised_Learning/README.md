# Semi-supervised Learning 

1. **MixMatch** (●)
2. **FixMatch** (●)
3. **FlexMatch** (●)  

--- 

1. Semi-supervised Learning이란? 
- 준지도 학습은 지도 학습과 비지도 학습의 조합으로 이루어진 학습이다. 
- 해당 학습은 레이블링된 데이터와 레이블링되지 않은 데이터가 모두 사용된다. 
- 준지도 학습의 특징은 한 쪽의 데이터에 있는 추가 정보를 활용해 다른 데이터 학습에서의 성능을 높이는 것을 목표로 한다. 
- 준지도 학습의 장점은 더 많은 데이터의 확보가 가능하다는 것이고 단점으로는 레이블링의 불확실성이 있다는 것이다. 
- 대표적인 방법으로는 Mixmatch, FixMatch, FlexMatch가 존재한다. 

<p align='center'><img src="./img/semi.jpg" width='500' height='300'></p>

**준지도 학습의 대표적인 방법론에 대한 한 줄 설명** 

2. [MixMatch](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Semi-supervised_Learning/pdf/MixMatch.pdf)
- 기존 준지도 학습 방법 Consistency Regularization, Entropy Minimization, Traditional Regularization (Mix Up)을 결합한 방법론 

3. [FixMatch](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Semi-supervised_Learning/pdf/FixMatch.pdf)
- MixMatch와 ReMixMatch의 경우 성능 고도화를 위해 주요 기법들을 추가 및 혼합하는 방향으로 발전함 
- 지나치게 정교한 loss term과 조정하기 어려운 수 많은 사용자 정의 파라미터를 사용하는 형태

4. [FlexMatch](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Semi-supervised_Learning/pdf/FlexMatch.pdf)
- 분류가 쉬운 범주의 경우 처음부터 Confidence가 높은 데이터가 다수 Pseudo Labeling이 되어 계속 더 잘 학습할 수 있게 유도되지만 분류가 어려운 범주는 Confidence가 높은 레이블이 없는 데이터가 많지 않기 때문에 비지도 학습의 본 의도인 레이블이 없는 데이터의 정보 활용이 어렵다는 문제가 존재
- 따라서, 모든 클래스에 동일한 Confidence 기준을 적용하지 않고 각 클래스의 난이도에 따른 다른 기준을 적용하는 것이 핵심! 


---

**Tutorial** 

- 본 튜터리얼에서는 MixMatch와 FixMatch 그리고 FlexMatch을 직접 구현하고 최종적으로 성능 비교를 해보려 한다. 실제 이론에서 성능은 FlexMatch > FixMatch > MixMatch로 알려져 있다. 따라서, 실제 이론과 부합하는지 확인하는 것(Top-1 ACC, Top-5 ACC)이 최종 목표이다. README 파일에서는 MixMatch에 관한 코드를 대표적으로 설명하고 FixMatch와 FlexMatch에 관한 코드는 업로드할 예정이다.

1. 활용 데이터 
- CIFAR-10 : 컬러 이미지들로 구성되어 있고 총 6만개의 샘플을 갖고 있고 그 중 5만개는 훈련을 위한 것이고 1만개는 테스트를 위한 것이다. CIFAR-10의 이미지들은 이름에서 나타나듯이 10개의 클래스에 속하고 해당 클래스는 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭으로 구성되어 있다.

<p align='center'><img src="./img/CIFAR-10.png" width='500' height='400'></p>

2. MixMatch 

- Hyperparamter Setting 
    - argparser라는 패키지를 이용해 사용하는 Hyperparameter를 저장한다. 
    - 대표적으로 Learning rate는 0.002 그리고 epoch수는 100으로 제한다. 

    ```
    def MixMatch_parser():
        parser = argparse.ArgumentParser(description="MixMatch PyTorch Implementation for BA")
        
        # method arguments
        parser.add_argument('--n-labeled', type=int, default=4000)
        parser.add_argument('--num-iter', type=int, default=1024,
                            help="The number of iteration per epoch")
        parser.add_argument('--alpha', type=float, default=0.75)
        parser.add_argument('--lambda-u', type=float, default=75)
        parser.add_argument('--T', default=0.5, type=float)
        parser.add_argument('--ema-decay', type=float, default=0.999)

        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.002)

        return parser
    ```

 - Define Function 

    
    ```
    # Custom Transform함수를 정의한다. 즉, 2가지의 Augmentation을 산출한다.
    class Transform_Twice:
        
        def __init__(self, transform):
            self.transform = transform
        
        def __call__(self, img):
            out1 = self.transform(img)
            out2 = self.transform(img)
            
            return out1, out2
    ```

    ```
    # Labeled data를 생성하는 함수

    class Labeled_CIFAR10(torchvision.datasets.CIFAR10):
        
        def __init__(self, root, indices=None,
                    train=True, transform=None,
                    target_transform=None, download=False):
            
            super(Labeled_CIFAR10, self).__init__(root,
                                            train=train,
                                            transform=transform,
                                            target_transform=target_transform,
                                            download=download)

            if indices is not None:
                self.data = self.data[indices]
                self.targets = np.array(self.targets)[indices]
            
            self.data = Transpose(Normalize(self.data))
        
        def __getitem__(self, index):
            
            img, target = self.data[index], self.targets[index]
            
            if self.transform is not None:
                img = self.transform(img)
            
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            return img, target
    ```

    ```
    # Unlabeled data를 생성하는 함수

    # Unlabeled data의 Label은 -1로 지정


    class Unlabeled_CIFAR10(Labeled_CIFAR10):
        
        def __init__(self, root, indices, train=True, transform=None, target_transform=None, download=False):
            
            super(Unlabeled_CIFAR10, self).__init__(root, indices, train,
                                                transform=transform,
                                                target_transform=target_transform,
                                                download=download)
            
            self.targets = np.array([-1 for i in range(len(self.targets))])
    ```

    ```
    # 데이터셋을 분할하기 위해서 Index를 섞는 함수 정의

    def split_datasets(labels, n_labeled_per_class):
        
        '''
        - n_labeled_per_class: labeled data의 개수
        - 클래스 내 500개 데이터는 validation data로 정의
        - 클래스 당 n_labeled_per_class 개수 만큼 labeled data로 정의
        - 나머지 이미지는 unlabeled data로 정의
        '''
        
        ### labeled, unlabeled, validation data 분할할 list 초기화
        labels = np.array(labels, dtype=int) 
        indice_labeled, indice_unlabeled, indice_val = [], [], [] 
        
        ### 각 class 단위로 loop 생성
        for i in range(10): 

            # 각각 labeled, unlabeled, validation data를 할당
            indice_tmp = np.where(labels==i)[0]
            
            indice_labeled.extend(indice_tmp[: n_labeled_per_class])
            indice_unlabeled.extend(indice_tmp[n_labeled_per_class: -500])
            indice_val.extend(indice_tmp[-500: ])
        
        ### 각 index를 Shuffle
        for i in [indice_labeled, indice_unlabeled, indice_val]:
            np.random.shuffle(i)
        
        return indice_labeled, indice_unlabeled, indice_val
    ```

    ```
    # CIFAR10에 대하여 labeled, unlabeled, validation, test dataset 생성

    def get_cifar10(data_dir: str, n_labeled: int,
                    transform_train=None, transform_val=None,
                    download=True):
        
        ### Torchvision에서 제공해주는 CIFAR10 dataset Download
        base_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=download)
        
        ### labeled, unlabeled, validation data에 해당하는 index를 가져오기
        indice_labeled, indice_unlabeled, indice_val = split_datasets(base_dataset.targets, int(n_labeled/10)) ### n_labeled는 아래 MixMatch_argparser 함수에서 정의
        
        ### index를 기반으로 dataset을 생성
        '''
        왜 unlabeled가 Transform_twice가 적용되었을까?
        '''
        train_labeled_set = Labeled_CIFAR10(data_dir, indice_labeled, train=True, transform=transform_train) 
        train_unlabeled_set = Unlabeled_CIFAR10(data_dir, indice_unlabeled, train=True, transform=Transform_Twice(transform_train))
        val_set = Labeled_CIFAR10(data_dir, indice_val, train=True, transform=transform_val, download=True) 
        test_set = Labeled_CIFAR10(data_dir, train=False, transform=transform_val, download=True) 

        return train_labeled_set, train_unlabeled_set, val_set, test_set
    ```

    ```
    # Image를 전처리 하기 위한 함수

    ### 데이터를 정규화 하기 위한 함수
    def Normalize(x, m=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2345, 0.2616)):
            
        ##### x, m, std를 각각 array화
        x, m, std = [np.array(a, np.float32) for a in (x, m, std)] 

        ##### 데이터 정규화
        x -= m * 255 
        x *= 1.0/(255*std)
        return x

    ### 데이터를 (B, C, H, W)로 수정해주기 위한 함수 (from torchvision.transforms 내 ToTensor 와 동일한 함수)
    def Transpose(x, source='NHWC', target='NCHW'):
        return x.transpose([source.index(d) for d in target])

    ### 특정 이미지에 동서남북 방향으로 4만큼 픽셀을 추가해주기 위한 학습
    def pad(x, border=4):
        return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')
    ```


    ```
    # Image를 Augmentation하기 위한 함수

    ### Image를 Padding 및 Crop적용

    1. object는 써도 되고 안써도 되는 것
    2. assert는 오류를 유도하기 위함 (나중에 이렇게 해놓으면 디버깅이 편함) --> 여기선 적절한 데이터 인풋의 형태를 유도

    class RandomPadandCrop(object):
        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
        
        def __call__(self, x):
            x = pad(x, 4)
            
            old_h, old_w = x.shape[1: ]
            new_h, new_w = self.output_size
            
            top = np.random.randint(0, old_h-new_h)
            left = np.random.randint(0, old_w-new_w)
            
            x = x[:, top:top+new_h, left:left+new_w]
            return x
        
        
    ### RandomFlip하는 함수 정의
    class RandomFlip(object):
        def __call__(self, x):
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            
            return x.copy()
        
        
    ### GaussianNoise를 추가하는 함수 정의
    class GaussianNoise(object):
        def __call__(self, x):
            c, h, w = x.shape
            x += np.random.randn(c, h, w)*0.15
            return x
    ```

    ```
    # Numpy를 Tensor로 변환하는 함수
    class ToTensor(object):
        def __call__(self, x):
            x = torch.from_numpy(x)
            return x
    ```
