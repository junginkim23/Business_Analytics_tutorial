## Kernel based Learning 

1. SVC vs SVR 
2. FDA (LDA) vs KFD 
3. PCA vs KPCA

## 1. SVC vs SVR 

`SVC` 
: 서포트 벡터 분류는 (SVC: Support Vector Classification) 분류 과제에 사용할 수 있는 강력한 머신러닝 지도학습 모델(SVM)을 말한다. 또 다른 표현으로는 분류를 위한 기준 선인 결정 경계를 정의하는 모델을 말한다. 따라서 결정 경계라는 걸 어떻게 정의하는지가 중요하다. 흔히, 두 개의 클래스가 존재하는 데이터를 분류하는 최적의 결정 경계는 두 클래스 사이에서 거리가 가장 먼 결정 경계를 말한다. 아래 그림에서 그림 F가 바로 최적의 결정 경계라고 볼 수 있다.

`what is a Support Vector?` 
: 결론 지어 말하면 결정 경계는 데이터 군으로부터 최대한 멀리 떨어지는 게 좋다는 것을 알 수 있다. 실제 Support Vector Machine에서 Support Vector는 결정 경계와 가까이 있는 데이터 포인트들을 말한다. 이를 사용하여 결정 경계를 정의하게 되는데 이 떄 알아야 할 용어가 바로 마진(Margin)이다.
<img src="./image/svm.png" width='90%' height='10%'>

`What is Margin?` 
: 마진은 결정 경계와 서포트 벡터 사이의 거리를 뜻한다. 아래 그림에서 실선이 결정 경계라면 실선과 점선간의 거리가 바로 마진이고 결국 마진을 최대화하는 결정 경계가 바로 최적의 결정 경계라고 볼 수 있다.

<img src="./image/margin.png" width='1000' height='400'>

[파이썬을 활용하여 SVR을 구현하여 확인해보았다.](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svc.ipynb)
- 본 tutorial에서는 load_breast_cancer() dataset을 사용해 선형, 비선형 분리를 진행해보았다. 비선형 분리 svm에서는 polynomial kernel을 사용한다.


`SVR`
: SVR (Support Vector Regression)은 회귀 문제에 사용할 수 있는 SVM 모델을 뜻한다. SVC에서는 마진을 최대화하는 결정 경계를 최적의 결정 경계로 설정하고 해당 결정 경계를 찾는 것을 목적으로 삼았지만 SVR에서는 마진 내부에 데이터가 최대한 많이 들어가도록 학습을 하는 것이다. 즉 회귀 계수 크기를 작게하여 회귀식을 평평하게 하되, 실제 값과 추정 값의 차이를 작도록 고려하는 회귀식을 찾는 것이 SVR의 목적이 된다. 

[파이썬을 활용하여 SVR을 구현하여 확인해보았다.](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svr.ipynb)
- 본 tutorial에서는 _를 사용해 선형, 비선형 회귀식을 통해 regression task를 해결해보았다.

## 2. FDA (LDA) vs KFD 

`FDA` 
: Fisher Discriminant Analysis (FDA)는 데이터들을 하나의 직선(1차원 공간)에 투영시킨 후 투영된 데이터들이 잘 구분이 되는지를 판단하는 방법으로 데이터들이 모여 있고, 중심부가 서로 멀수록 데이터의 구분이 잘 되었다고 판단한다. 

`KFD` 
: Kernel Fisher Discriminant Analysis (KFD)는 기존에 존재한 FDA에 kernel화를 진행한 것만 차이가 있다. 따라서 선형 분리가 불가능한 입력 샘플 데이터를 kernel function을 사용하여 고차원 공간으로 변환한 뒤 선형 분리가 가능하게 끔 도와주는 kernel trick이 입력 샘플에 적용된 것 이외에 기존 FDA와 동일하다.