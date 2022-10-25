## Kernel based Learning 

1. SVC vs SVR 
2. FDA (LDA) vs KFD 
3. PCA vs KPCA

## 1. SVC vs SVR 

`SVC` 
: 서포트 벡터 분류는 (SVC: Support Vector Classification) 분류 과제에 사용할 수 있는 강력한 머신러닝 지도학습 모델(SVM)을 말한다. 또 다른 표현으로는 분류를 위한 기준 선인 결정 경계를 정의하는 모델을 말한다. 따라서 결정 경계라는 걸 어떻게 정의하는지가 중요하다. 흔히, 두 개의 클래스가 존재하는 데이터를 분류하는 최적의 결정 경계는 두 클래스 사이에서 거리가 가장 먼 결정 경계를 말한다. 아래 그림에서 그림 F가 바로 최적의 결정 경계라고 볼 수 있다.

`what is a Support Vector?` 
: 결론 지어 말하면 결정 경계는 데이터 군으로부터 최대한 멀리 떨어지는 게 좋다는 것을 알 수 있다. 실제 Support Vector Machine에서 Support Vector는 결정 경계와 가까이 있는 데이터 포인트들을 말한다. 이를 사용하여 결정 경계를 정의하게 되는데 이 떄 알아야 할 용어가 바로 마진(Margin)이다.
<img src="./image/svm.png" width='1000' height='300'>

`What is Margin?` 
: 마진은 결정 경계와 서포트 벡터 사이의 거리를 뜻한다. 아래 그림에서 실선이 결정 경계라면 실선과 점선간의 거리가 바로 마진이고 결국 마진을 최대화하는 결정 경계가 바로 최적의 결정 경계라고 볼 수 있다.

<img src="./image/margin.png" width='800' height='300'>

[SVC - Python Tutorial](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svc.ipynb)
- sklearn.datasets에 있는 load_breast_cancer dataset을 사용하였다. 
```
import sklearn.datasets as d 

x = d.load_breast_cancer()

print(x.DESCR)
Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        The mean, standard error, and "worst" or largest (mean of the three
        worst/largest values) of these features were computed for each image,
...
     July-August 1995.
   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
     163-171.
```
- 선형, 비선형 분리 진행 
```
## linear로 선형 분리 진행 
svm_clf = svm.SVC(kernel='linear')

## 교차 검증 진행 
print(pd.DataFrame(cross_validate(svm_clf,X,y,cv=6)))

print('교차 검증 결과 평균:', cross_val_score(svm_clf,X,y,cv=6).mean())

fit_time  score_time  test_score
0  0.509360    0.001000    0.968421
1  1.618746    0.001000    0.905263
2  0.568252    0.002030    0.957895
3  1.022619    0.001005    0.957895
4  0.308602    0.001004    0.936842
5  0.682040    0.001537    0.968085
교차 검증 결과 평균: 0.9490668159761105

## Kernel trick (polynomial kernel) 사용해 비선형 분리 진행 
svm_clf_poly=svm.SVC(kernel='poly')

## 교차 검증 진행
print(pd.DataFrame(cross_validate(svm_clf_poly,X,y,cv=6)))

print('교차 검증 결과 평균:', cross_val_score(svm_clf_poly,X,y,cv=6).mean())

fit_time  score_time  test_score
0  0.003999    0.000972    0.863158
1  0.003028    0.002000    0.842105
2  0.002970    0.001918    0.926316
3  0.002999    0.000999    0.936842
4  0.002999    0.000973    0.957895
5  0.003000    0.001000    0.925532
교차 검증 결과 평균: 0.9086412840612169
```
- optuna를 사용해 선형 분리에서 적절한 하이퍼 파라미터 C값 탐색 진행 
```
## 선형 분리에서 적절한 C값 탐색 진행
warnings.filterwarnings('ignore')

def objective(trial):
    x = d.load_breast_cancer()
    X,y = x.data, x.target 

    svc_c = trial.suggest_float('C',1e-4,1e2,log=True)
    classifier_obj = svm.SVC(C=svc_c, gamma='auto',kernel='linear')
    
    X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(X,y,random_state=0)
    classifier_obj.fit(X_train,y_train)
    y_pred = classifier_obj.predict(X_val)

    accuracy = sklearn.metrics.accuracy_score(y_val,y_pred,normalize=True)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=100)

print('100번의 trial 중 최적의 하이퍼 파라미터:',study.best_trial.params)
print('100번의 trial 중 가장 높은 accuracy:',study.best_trial.value)
---output---
100번의 trial 중 최적의 하이퍼 파라미터: {'C': 36.60317395134434}
100번의 trial 중 가장 높은 accuracy: 0.972027972027972
```

- Kernel trick(polynomial)을 사용한 svc에서 optuna를 사용해 최적의 하이퍼 파라미터 탐색  
```
## 비선형 분리에서 적절한 C값 탐색 진행
warnings.filterwarnings('ignore')

def objective_poly(trial):
    x = d.load_breast_cancer()
    X,y = x.data, x.target 

    svc_c = trial.suggest_float('C',1e-4,1e-2,log=True)
    svc_degree = trial.suggest_int('degree',3,5,step=1)
    # svc_gamma = trial.suggest_categorical('svc_gamma',['scale','auto'])
    svc_coef = trial.suggest_float('coef0',0.0,0.3,step=0.1)

    classifier_obj_non = svm.SVC(C=svc_c, gamma='auto',kernel='poly')
    
    X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(X,y,random_state=0)
    classifier_obj_non.fit(X_train,y_train)
    y_pred = classifier_obj_non.predict(X_val)

    accuracy = sklearn.metrics.accuracy_score(y_val,y_pred,normalize=True)
    return accuracy

study_non = optuna.create_study(direction='maximize')
study_non.optimize(objective_poly,n_trials=100)

print('100번의 trial 중 최적의 하이퍼 파라미터:',study.best_trial.params)
print('100번의 trial 중 가장 높은 accuracy:',study.best_trial.value)
```

- plot 그리기 
```
from sklearn.decomposition import PCA

model_best_params = study.best_params
model_best_params['C'] = model_best_params.pop('svc_c')
model_best_params['kernel'] = 'linear'
model_best_params['gamma'] = 'auto'

model = svm.SVC(**model_best_params)
pca = PCA(n_components=2)
pca_X = pca.fit_transform(X)

model.fit(pca_X,y)

plot_decision_regions(X=pca_X,y=y.values, clf=model, legend=2)
plt.show()
```
<p align='center'><img src="./image/output.png" width='400' height='300'></p>


`SVR`
: SVR (Support Vector Regression)은 회귀 문제에 사용할 수 있는 SVM 모델을 뜻한다. SVC에서는 마진을 최대화하는 결정 경계를 최적의 결정 경계로 설정하고 해당 결정 경계를 찾는 것을 목적으로 삼았지만 SVR에서는 마진 내부에 데이터가 최대한 많이 들어가도록 학습을 하는 것이다. 즉 회귀 계수 크기를 작게하여 회귀식을 평평하게 하되, 실제 값과 추정 값의 차이를 작도록 고려하는 회귀식을 찾는 것이 SVR의 목적이 된다. 

[파이썬을 활용하여 SVR을 구현하여 확인해보았다.](https://github.com/junginkim23/Business_Analytics_tutorial/blob/master/Kernel_based_learning/svr.ipynb)
- 본 tutorial에서는 _를 사용해 선형, 비선형 회귀식을 통해 regression task를 해결해보았다.

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