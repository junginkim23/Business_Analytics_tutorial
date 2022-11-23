# Ensemble

1. Ensemble
2. **Bagging & random Forest** (●)
3. **Boosting-based Ensemble** (●)

---

1. 앙상블이란? 
- 어떤 데이터의 값을 예측할 때, 주로 하나의 모델만을 사용한다. 만약, 여러 개의 모델을 조화롭게 학습시켜 해당 모델들의 예측 결과를 이용한다면 더 정확한 예측값을 구할 수 있을 것이다라는 생각에서부터 앙상블은 시작된다. 

- 앙상블 학습에서의 핵심은 여러 개의 약 분류기(Weak Learner)를 결합해 하나의 강 분류기(Strong Learner)를 만드는 것이다. 

- 앙상블 학습은 크게 Bagging & Boosting 두 가지로 나뉜다. 

<p align='center'><img src="./img/total.jpg" width='700' height='300'></p>

2-1. Bagging 
- Bagging은 Booststrap Aggregation의 약자로 한 데이터 셋에서 샘플을 복원 추출하여 여러 데이터 셋인 Bootstrap을 만들어 각 모델을 학습시켜 결과물을 집계(Aggregation)하여 최종 결과 값을 만드는 방법이다. 

- 처음 Bagging을 접하게 되면 복원 추출이라는 말이 생소하게 다가올 수 있는데, 복원 추출과 비교해서 살펴볼 용어는 비복원 추출이다. 복원 추출은 한 번 시행한 결과를 다시 얻을 수 있도록 모집단에 다시 포함시켜 추출하는 방법이고 비복원 추출은 그 반대를 뜻한다. 

- 모델의 결과가 만약 범주 형태라면 투표 방식(Voting)으로 결과를 집계하고, 연속적인 수치형 데이터라면 평균으로 집계를 진행한다. 

- 이번 tutorial에서는 bagging하면 대표적인 방법인 random forest를 사용할 것이다.

<p align='center'><img src="./img/bagging2.jpg" width='500' height='300'></p>

2-2. Random Forest
- 랜덤 포레스트는 과적합을 방지하기 위해서 최적의 기준 변수를 임의로 선택하는 머신러닝 기법으로 여러 개의 의사결정나무를 만들고 그 모습이 마치 숲을 이루는 것과 같다고 하여 Random Forest라는 이름이 붙었다. 

- Bootstrapping을 통해 데이터 셋을 형성할 때 선택되지 않은 변수들이 있다면 해당 변수들을 모은 집합은 Out of Bag (OOB)로 설정하여 validation을 진행할 때 사용한다.

<p align='center'><img src="./img/randomforest.jpg" width='700' height='300'></p>

3-1. Boosting 

- Boosting도 마찬가지로 여러 약한 분류기가 합쳐져 강한 분류기를 사용해 최종 모델로 사용한다. 

- 즉, 여러 개의 모델이 존재할 때 순차적으로 학습-예측을 진행하면서 이전에 학습된 알고리즘의 예측이 틀린 데이터를 올바르게 예측할 수 있도록 그 다음 알고리즘에 가중치를 부여하여 학습과 예측을 진행하는 방법이다. 

<p align='center'><img src="./img/boosting.jpg" width='500' height='400'></p>

3-2. XGBoost, LightGBM, CatBoost

- Boosting의 대표적인 방법으로 Gradient Boosting Machine (GBM)이 존재한다. 그리고 GBM의 단점을 개선한 여러 모델이 XGBoost, LightGBM, CatBoost가 있다. 먼저 GBM은 약한 학습기를 잔차 자체에 적합하고, 이전의 예측값에 예측한 잔차를 더해 주어 해당 값을 다음 모델의 target으로 삼아 학습을 진행한다. 결국 최종 강한 분류기에서 나오는 예측값을 초기화 모형과 더해 최종 결과값으로 산출한다.

<p align='center'><img src="./img/gbm.jpg" width='700' height='400'></p>

- 이러한 GBM은 과적합 문제, 느린 속도와 같은 여러 문제가 존재한다. XGBoost는 이러한 문제점을 보완해 GBM보다 빠른 속도, 과적합 방지를 규제하기 위한 파라미터가 존재한다. 

- LightGBM의 경우 기존 XGBoost와의 차이점은 트리 분할 방식에 있다. 먼저 XGBoost의 경우 균형 트리 분할 방식을 사용하고 LightGBM의 경우 리프 중심 트리 분할 방식을 사용한다. 이러한 리프 중심 트리 분할 방식은 균형 트리 분할 방식에 비해 예측 오류 손실을 최소화할 수 있다는 장점이 존재한다. 

- 마지막으로 CatBoost는 이름에서 유추할 수 있듯이 Cat, category 즉, 범주형 변수가 많은 데이터를 학습할 때 성능이 좋다는 장점이 있다. 특히, XGBoost와 LightGBM에 비해 학습 속도가 더 빠르다는 장점 또한 존재한다. 그리고 앞서 설명했던 두 방법에 비해 하이퍼 파라미터에 덜 민감하다는 장점이 있다. 해당 방식의 경우 XGBoost와 동일하게 균형 트리 분할 방식을 사용한다.

4. 배깅과 부스팅의 차이 
- 배깅은 병렬로 학습이 되는 반면에 부스팅은 순차적으로 학습하는 것이 가장 큰 차이점이다. 


- 즉, 부스팅은 한 번 학습이 끝나면 오답에 대해서는 높은 가중치를 부여하고 정답에 대해서는 낮은 가중치를 부여한다. 오답에 더 집중할 수 있게 해주는 역할을 가중치를 활용해 하게 되는 것이다. 

- 다만, 부스팅은 배깅에 비해 에러가 굉장히 적고 성능이 좋지만 속도가 상대적으로 느리고 과적합될 가능성이 크다. 따라서 상황에 따른 적절한 방법을 선택하여 사용하는 것이 중요하다.

**튜토리얼을 진행하기에 앞서 앙상블이 무엇이고 앙상블에 대표적인 두 방식은 bagging과 boosting을 살펴보았다. 그리고 각 방식에 존재하는 대표적인 방법들에 대해서도 간략하게 알아 보았다. 위에서 언급한 내용들은 간략한 소개이기 때문에 좀 더 정확한 내용을 듣고 싶다면 참조 링크들에서 확인 가능하다.**

---

**Tutorial**

**Reference**
- swallow.github.io
- https://julie-tech.tistory.com/119
- https://tyami.github.io/machine%20learning/ensemble-1-basics/
- https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-11-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5-Ensemble-Learning-%EB%B0%B0%EA%B9%85Bagging%EA%B3%BC-%EB%B6%80%EC%8A%A4%ED%8C%85Boosting
- https://assaeunji.github.io/machine%20learning/2020-09-05-gbm/
- https://bioinformaticsandme.tistory.com/167