## Kernel based Learning 

1. SVM vs SVR 
2. FDA (LDA) vs KFD 
3. PCA vs KPCA

## 1. SVM vs SVR 

*SVM* 
: 서포트 벡터 머신은 (SVM: Support Vector Machine) 분류 과제에 사용할 수 있는 강력한 머신러닝 지도학습 모델을 말한다. 또 다른 표현으로는 분류를 위한 기준 선인 결정 경계를 정의하는 모델을 말한다. 따라서 결정 경계라는 걸 어떻게 정의하는지가 중요하다. 흔히, 두 개의 클래스가 존재하는 데이터를 분류하는 최적의 결정 경계는 두 클래스 사이에서 거리가 가장 먼 결정 경계를 말한다. 아래 그림에서 그림 F가 바로 최적의 결정 경계라고 볼 수 있다.

*what is a Support Vector?* 
: 결론 지어 말하면 결정 경계는 데이터 군으로부터 최대한 멀리 떨어지는 게 좋다는 것을 알 수 있다. 실제 Support Vector Machine에서 Support Vector는 결정 경계와 가까이 있는 데이터 포인트들을 말한다. 이를 사용하여 결정 경계를 정의하게 되는데 이 떄 알아야 할 용어가 바로 마진(Margin)이다.
<img src="./image/svm.png" width='100%' height='10%'>

*What is Margin?* 
: 마진은 결정 경계와 서포트 벡터 사이의 거리를 뜻한다. 아래 그림에서 실선이 결정 경계라면 실선과 점선간의 거리가 바로 마진이고 결국 마진을 최대화하는 결정 경계가 바로 최적의 결정 경계라고 볼 수 있다.

<img src="./image/margin.png" width='50%' height='10%'>

- [파이썬을 활용하여 svm을 구현하여 확인해보았다.](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svm.ipynb) 해당 링크를 통해 들어가게 되면 soft margin과 hard margin에 대한 간략한 설명도 확인 가능하다.