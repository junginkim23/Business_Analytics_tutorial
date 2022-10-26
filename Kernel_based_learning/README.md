# Kernel based Learning 

1. SVC vs SVR 
2. FDA (LDA) vs KFD 
3. PCA vs KPCA

---

## 1. SVC vs SVR 

`SVC` 
: Support Vector Classification (SVC) refers to a powerful machine learning supervised learning model (SVM) that can be used for classification tasks. Another expression is a model that defines a decision boundary that is a baseline for classification. Therefore, it is important how we define decision boundaries. Often, the optimal decision boundary for classifying data in which two classes exist is the decision boundary with the longest distance between the two classes. In the figure below, Figure F can be seen as the optimal decision boundary.

`what is a Support Vector?` 
: In conclusion, it can be seen that the decision boundary should be as far away from the data set as possible. In a real support vector machine, the support vector refers to the data points that are close to the decision boundary. This is used to define the decision boundary, and the term you need to know at this time is the margin.
<p align='center'><img src="./image/svm.png" width='400' height='300'></p>

`What is Margin?` 
: Margin is the distance between the decision boundary and the support vector. In the figure below, if the solid line is the decision boundary, the distance between the solid line and the dotted line is the margin, and the decision boundary that maximizes the margin is the optimal decision boundary.

<p align='center'><img src="./image/margin.png" width='400' height='300'></p>

[SVC - Python Tutorial](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svc.py)

`SVR`
: SVR (Support Vector Regression)은 회귀 문제에 사용할 수 있는 SVM 모델을 뜻한다. SVC에서는 마진을 최대화하는 결정 경계를 최적의 결정 경계로 설정하고 해당 결정 경계를 찾는 것을 목적으로 삼았지만 SVR에서는 마진 내부에 데이터가 최대한 많이 들어가도록 학습을 하는 것이다. 즉 회귀 계수 크기를 작게하여 회귀식을 평평하게 하되, 실제 값과 추정 값의 차이를 작도록 고려하는 회귀식을 찾는 것이 SVR의 목적이 된다. 

[SVR - Python Tutorial](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svr.py)

## 2. FDA (LDA) vs KFD 

`FDA` 
: Fisher Discriminant Analysis (FDA)는 데이터들을 하나의 직선(1차원 공간)에 투영시킨 후 투영된 데이터들이 잘 구분이 되는지를 판단하는 방법으로 데이터들이 모여 있고, 중심부가 서로 멀수록 데이터의 구분이 잘 되었다고 판단한다. 

`KFD` 
: Kernel Fisher Discriminant Analysis (KFD)는 기존에 존재한 FDA에 kernel화를 진행한 것만 차이가 있다. 따라서 선형 분리가 불가능한 입력 샘플 데이터를 kernel function을 사용하여 고차원 공간으로 변환한 뒤 선형 분리가 가능하게 끔 도와주는 kernel trick이 입력 샘플에 적용된 것 이외에 기존 FDA와 동일하다.

## 3. PCA vs KPCA 

`PCA` 
: 주성분 분석이라 불리는 PCA는 차원 축소와 변수 추출 기법으로 널리 쓰이고 있다. 여기서 주성분이란 전체 독립 변수들의 분산을 가장 잘 설명하는 성분을 말한다. 해당 기법은 변수가 너무 많아 기존 변수 조합을 통해 새로운 변수를 만들어 낼 때 주로 사용한다. PCA는 실제로 분산을 최대한으로 보존하는 축을 선택하게 되는데 이러한 이유는 분산이 커져야 데이터들 사이의 차이점이 명확해지고 정보를 가장 적게 손실하기 때문이다. 

`KPCA`
: KPCA는 실제로 PCA에 Kernel trick을 적용하는 것이 PCA와의 차이점인데, 선형 분리가 불가능한 샘플 혹은 데이터 셋에 Kernel function을 사용하여 고차원으로 변환한 이후 linear PCA를 사용하는 것을 말한다. 