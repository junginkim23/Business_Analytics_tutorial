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

- *XGBoost* : 이러한 GBM은 과적합 문제, 느린 속도와 같은 여러 문제가 존재한다. *XGBoost*는 이러한 문제점을 보완해 GBM보다 빠른 속도, 과적합 방지를 규제하기 위한 파라미터가 존재한다. 

- *LightGBM* : *LightGBM*의 경우 기존 XGBoost와의 차이점은 트리 분할 방식에 있다. 먼저 *XGBoost*의 경우 균형 트리 분할 방식을 사용하고 *LightGBM*의 경우 리프 중심 트리 분할 방식을 사용한다. 이러한 리프 중심 트리 분할 방식은 균형 트리 분할 방식에 비해 예측 오류 손실을 최소화할 수 있다는 장점이 존재한다. 

- *CatBoost* : 마지막으로 *CatBoost*는 이름에서 유추할 수 있듯이 Cat, category 즉, 범주형 변수가 많은 데이터를 학습할 때 성능이 좋다는 장점이 있다. 특히, XGBoost와 LightGBM에 비해 학습 속도가 더 빠르다는 장점 또한 존재한다. 그리고 앞서 설명했던 두 방법에 비해 하이퍼 파라미터에 덜 민감하다는 장점이 있다. 해당 방식의 경우 XGBoost와 동일하게 균형 트리 분할 방식을 사용한다.

4. 배깅과 부스팅의 차이 
- 배깅은 병렬로 학습이 되는 반면에 부스팅은 순차적으로 학습하는 것이 가장 큰 차이점이다. 

- 즉, 부스팅은 한 번 학습이 끝나면 오답에 대해서는 높은 가중치를 부여하고 정답에 대해서는 낮은 가중치를 부여한다. 오답에 더 집중할 수 있게 해주는 역할을 가중치를 활용해 하게 되는 것이다. 

- 다만, 부스팅은 배깅에 비해 에러가 굉장히 적고 성능이 좋지만 속도가 상대적으로 느리고 과적합될 가능성이 크다. 따라서 상황에 따른 적절한 방법을 선택하여 사용하는 것이 중요하다.

**튜토리얼을 진행하기에 앞서 앙상블이 무엇이고 앙상블에 대표적인 두 방식은 bagging과 boosting을 살펴보았다. 그리고 각 방식에 존재하는 대표적인 방법들에 대해서도 간략하게 알아 보았다. 위에서 언급한 내용들은 간략한 소개이기 때문에 좀 더 정확한 내용을 듣고 싶다면 참조 링크들에서 확인 가능하다.**

---

**Tutorial**

- **개요**
    - 이번 튜토리얼에서 사용할 데이터는 심뇌혈관 질환 발생빈도에 대한 데이터(일별, 성별, 시/도별)와 각 시/도별 대표지역의 예보 데이터(직접 전처리 진행)이다. 

    - 해당 데이터를 통해 최종적으로 예측하고자 하는 대상은 지역별 심뇌혈관 발생빈도이다. 

    - 본 튜터리얼에서는 Ensemble 기법 중 Bagging, Boosting의 대표적인 방법인 Randomforest, XGBoost, LightGBM을 사용하고 추가로 Lidge와 Lasso까지 결합한 stacking 기법까지 적용하여 결과를 확인한다.

    - 학습을 진행할 때 사용되는 변수의 수가 많으므로 VIF를 사용해 다중 공산성을 띄는 변수를 제거한 데이터와 전체 변수를 사용하는 데이터 모두 사용해 두 가지 형태의 데이터로 학습을 진행한다. 

    - 평가 지표로는 MSE, MAE, R2-score를 사용한다.

    - 마지막으로 사후분석에 자주 사용되는 explainable AI(XAI) 방법론의 하나인 SHAP를 사용하여 모델이 중요하게 보는 변수와 해당 변수의 영향력에 대한 분석을 진행한다. 


- Shell script를 사용해 두 가지 데이터 형태를 사용했을 때의 결과를 한 번에 확인한다. 
```
test_mode =('vif' 'all')

for mode_1 in ${test_mode[@]}
do
    python main.py --mode_1 $mode_1
done
```

- main문은 아래와 같다.
```
def main():

    warnings.filterwarnings('ignore')

    args = get_args()

    X_train, X_test, y_train, y_test = MakeDataset(args).make_data()
    pred_result_1 = []
    pred_result_2 = []

    if args.mode_1 == 'vif': 
        study_xgb = XGBoost(X_train,y_train,args.trials).tuning()
        study_lgb = LightGBM(X_train,y_train,args.trials).tuning()
        study_rf = RandomForest(X_train,y_train,args.trials).tuning()
        model_list_1 = [study_lgb,study_xgb,study_rf]
        stacking = Stacking(X_train,y_train,model_list_1).stack()
        
        pred_result_1.append(Tester(X_train,y_train,X_test,y_test).test(model_list_1))
        pred_result_1.append(stacking.predict(X_test))

        Tester(X_train,y_train,X_test,y_test).compute_score(pred_result_1)   

    else: 
        study_xgb = XGBoost(X_train,y_train,args.trials).tuning()
        study_lgb = LightGBM(X_train,y_train,args.trials).tuning()
        study_rf = RandomForest(X_train,y_train,args.trials).tuning()
        model_list_2 = [study_lgb,study_xgb,study_rf]
        stacking = Stacking(X_train,y_train,model_list_2).stack()

        pred_result_2.append(Tester(X_train,y_train,X_test,y_test).test(model_list_2))
        pred_result_2.append(stacking.predict(X_test))

        Tester(X_train,y_train,X_test,y_test).compute_score(pred_result_2)
```

- **데이터** 

    - 전체 변수를 사용하는 데이터와 vif를 사용해 다중 공산성을 띄는 변수를 제거한 데이터를 사용해 예측 모델(XGBoost, LightGBM, RandomForest, Stacking)의 성능을 확인한다.

    - [전체 데이터] 
        - 학습에 사용되는 데이터의 수는 총 39739개이며 변수의 수는 총 48개이다. 
        - 검증에 사용되는 데이터의 수는 총 9935개이며 변수의 수는 같다.
        - 변수는 지역명, 성별, 최고 기온, 최저 기온, 평균 강수량, 체감 온도, 총인구수, 나이대별 인구수, 예보 데이터(풍속, 풍향, 기온..) 등이 사용된다. 


    - [Vif를 사용해 변수를 제거한 데이터]
        - 독립변수의 수가 48개로 많다 보니 독립변수 간 상관관계가 심한 변수는 제외하기 위해 VIF를 사용하여 독립 변수 간 상관관계의 척도를 측정한다. 
        - 간단하게 말하면, VIF는 다중 회귀 모델에서 독립 변수 간 상관관계 유무를 파악하는 척도로서 VIF 수치가 10이 넘으면 다중 공산성이 있다고 판단하고 5가 넘으면 주의 수준으로 판단한다. 
        - vif를 통해 제거된 변수는 총 20개('10대_인구수','총_인구수','40대_인구수','avgTa','20대_인구수','예보_3시간기온','60대_인구수','90대_인구수','체감온도','avgTd','30대_인구수','예보_일최저기온','70대_인구수','minTa','avgTs','50대_인구수','80대_인구수','minTg','maxTa','avgPv')이고 남은 변수 28개의 변수로 학습을 진행한다.
        - 학습 데이터와 검증 데이터의 수는 전체 데이터를 사용하였을 때와 같다.

    <p align='center'><img src="./img/data.jpg" width='1500' height='150'></p>

    1. 전처리 
        - 먼저, 결측치를 대체하거나 제거하는 작업을 시행한다. 
        - 특히, 세종시의 경우 2012년 7월에 설립되었기 때문에 1월~6월 사이의 데이터는 존재하지 않는다. 이러면 세종시에 남아 있는 데이터는 2012년 7월 이후 데이터를 기준으로 평균값으로 결측치를 대체한다. 
        - 이외에 지역별로 결측값이 존재한다면 주변 지역의 데이터를 사용해 K-means를 사용해 결측값을 대체한다. 
        - 모델 학습의 입력으로 사용하기 위해서 지역명과 같은 object 형태의 데이터는 labelencoder를 사용해 encoding을 진행한다. 


    ```
  class MakeDataset():

    def __init__(self,args):

        self.args = args
        self.le = LabelEncoder()

    def make_data(self):
        # Simple preprocessing and standardization of train data

        data = pd.read_csv(self.args.data_path) # Replacing Missing values using k-means
        X_data = data.drop(['freq'],axis = 1)
        y_data = data['freq']

        # Used to encode stnNm features
        self.le.fit(X_data['stnNm'])
        X_data['stnNm'] = self.le.transform(X_data['stnNm'])

        # Converting to an integer type of feature related to the population 
        for col in X_data.columns:
            if X_data[col].dtype == 'object':
                X_data[col] = X_data[col].apply(lambda x : x.replace(',',''))
                X_data[col] = X_data[col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.2, random_state=328)

        if self.args.mode_1 == 'vif':
            X_train,X_test = VIF(X_train, X_test).after_vif()
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test
    ```

    ```
   def after_vif(self):
        for _ in range(1000):
    
            vif = self.check_vif()

            if len(vif[vif['VIF Factor'] >=10]) >=1:
                vif_value = round(vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[0,0],3)
                remove_col = vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[0,1]

                if (remove_col == 'const') & (vif[vif['VIF Factor'] >=10].shape[0] == 1):
                    print('VIF가 10이 넘는 변수가 없습니다. === FOR LOOP을 종료합니다.')
                    break

                elif (remove_col == 'const') & (vif[vif['VIF Factor'] >=10].shape[0] != 1):
                    vif_value = round(vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[1,0],3)
                    remove_col = vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[1,1]
                    self.remove_list.append(remove_col)
                    self.X.drop(remove_col, axis=1, inplace=True)
                    t2 = time.time()
                    elapsed_time = t2-self.t
                    print('VIF값이 '+str(vif_value)+'인 '+remove_col+'이 제거되었습니다. === 현재 총 제거된 변수의 개수는 '+str(len(self.remove_list))+'개 입니다. === 경과된 시간: '+str(round(elapsed_time/60))+'분')

                else:
                    self.remove_list.append(remove_col)
                    self.X.drop(remove_col, axis=1, inplace=True)
                    t2 = time.time()
                    elapsed_time = t2-self.t
                    print('VIF값이 '+str(vif_value)+'인 '+remove_col+'이 제거되었습니다. === 현재 총 제거된 변수의 개수는 '+str(len(self.remove_list))+'개 입니다. === 경과된 시간: '+str(round(elapsed_time/60))+'분')


            else:
                print('VIF가 10이 넘는 변수가 없습니다. === FOR LOOP을 종료합니다.')
                break

            
        vif_col = self.X.columns.to_list()
        test_vif = self.X_test[vif_col]

        print(vif_col)

        return self.X, test_vif

    ### output 
    VIF값이 inf인 10대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 1개 입니다. === 경과된 시간: 0분
    VIF값이 40779.5인 총_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 2개 입니다. === 경과된 시간: 0분
    VIF값이 1947.383인 40대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 3개 입니다. === 경과된 시간: 0분
    VIF값이 1871.255인 avgTa이 제거되었습니다. === 현재 총 제거된 변수의 개수는 4개 입니다. === 경과된 시간: 0분
    VIF값이 1400.256인 20대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 5개 입니다. === 경과된 시간: 0분
    VIF값이 1300.126인 예보_3시간기온이 제거되었습니다. === 현재 총 제거된 변수의 개수는 6개 입니다. === 경과된 시간: 0분
    VIF값이 651.233인 60대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 7개 입니다. === 경과된 시간: 0분
    VIF값이 421.694인 90대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 8개 입니다. === 경과된 시간: 0분
    VIF값이 412.492인 체감온도이 제거되었습니다. === 현재 총 제거된 변수의 개수는 9개 입니다. === 경과된 시간: 0분
    VIF값이 277.595인 avgTd이 제거되었습니다. === 현재 총 제거된 변수의 개수는 10개 입니다. === 경과된 시간: 0분
    VIF값이 176.414인 30대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 11개 입니다. === 경과된 시간: 1분
    VIF값이 104.127인 예보_일최저기온이 제거되었습니다. === 현재 총 제거된 변수의 개수는 12개 입니다. === 경과된 시간: 1분
    VIF값이 97.605인 70대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 13개 입니다. === 경과된 시간: 1분
    VIF값이 94.824인 minTa이 제거되었습니다. === 현재 총 제거된 변수의 개수는 14개 입니다. === 경과된 시간: 1분
    VIF값이 40.373인 avgTs이 제거되었습니다. === 현재 총 제거된 변수의 개수는 15개 입니다. === 경과된 시간: 1분
    VIF값이 33.283인 50대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 16개 입니다. === 경과된 시간: 1분
    VIF값이 26.77인 80대_인구수이 제거되었습니다. === 현재 총 제거된 변수의 개수는 17개 입니다. === 경과된 시간: 1분
    VIF값이 26.576인 minTg이 제거되었습니다. === 현재 총 제거된 변수의 개수는 18개 입니다. === 경과된 시간: 1분
    VIF값이 21.878인 maxTa이 제거되었습니다. === 현재 총 제거된 변수의 개수는 19개 입니다. === 경과된 시간: 1분
    VIF값이 11.304인 avgPv이 제거되었습니다. === 현재 총 제거된 변수의 개수는 20개 입니다. === 경과된 시간: 1분
    VIF가 10이 넘는 변수가 없습니다. === FOR LOOP을 종료합니다.

    vif를 사용해서 제거하고 남은 변수 : 
    ['stnNm', 'tm', 'sex', 'sumRn', 'maxWs', 'avgWs', 'minRhm', 'avgRhm', 'avgPa', 'avgPs', '10세이하_인구수', '100세이상_인구수', 'SPI1', 'SPI2', 'SPI3', 'SPI4', 'SPI5', 'SPI6', 'SPI9', 'SPI12', 'SPI18', 'SPI24', '미세먼지농도', '예보_강수확률', '예보_습도', '예보_일최고기온', '예보_풍속', '예보_풍향']
    ```

    2. 모델링
        - 학습을 위해 사용한 모델은 XGBoost, Random Forest, LightGBM, Stacking(XGBoost, LightGBM, Ridge, Lasso)이다. 
        - 데이터별 모델별 성능 확인 이후 최종 가장 좋은 성능을 나타내는 모델에 대해 shap 방법론을 사용해 사후 분석을 진행할 예정이다. 
        - 각 모델의 하이퍼 파라미터 튜닝을 위해 optuna 기법을 적용해서 최적의 파라미터를 산출하는 것도 포함한다.
        - 각 모델의 튜닝 파라미터는 어느 정도 예측에 높은 중요도를 갖는 파라미터들을 선정한다. 
        - 최적의 파라미터를 찾을 때 rmse의 값이 최소가 되는 파라미터를 탐색한다. 
        - 모델별 최적의 value를 가지는 최적의 파라미터는 아래의 코드에 포함되어 있다.
        - 20번의 trial 중에서 가장 좋은 value 뽑은 결과 미묘하지만 모든 변수를 사용했을 때 조금 더 좋은 결과를 뽑는 것을 확인할 수 있었다.

        ```
        1. RandomForest
        class RandomForest():
        def __init__(self,X_train,y_train,trials):
            super(RandomForest,self).__init__()
            self.X = X_train
            self.y = y_train
            self.trials = trials

        def objective(self,trial):
            rf_estimators = trial.suggest_int('n_estimators', 100, 300)
            rf_depth = trial.suggest_int('max_depth', 3,8)
            rf_samples_leaf =  trial.suggest_int('min_samples_leaf',3,15)
            rf_samples_split =  trial.suggest_int('min_samples_split',2,10)
            rf_features =  trial.suggest_categorical('max_features', ['sqrt','log2'])

            regressor_obj = RandomForestRegressor(n_estimators=rf_estimators,max_depth=rf_depth,
                                                max_features=rf_features,
                                                min_samples_leaf=rf_samples_leaf,
                                                min_samples_split=rf_samples_split)

            rmse = np.sqrt(-cross_val_score(regressor_obj, self.X, self.y, scoring="neg_mean_squared_error", cv = 10, n_jobs=8))
            rmse = rmse.min()   
            return rmse

        def tuning(self):
            sampler = TPESampler(seed=42) # TPESampler 

            study_rf = optuna.create_study(direction='minimize', sampler=sampler)
            study_rf.optimize(self.objective, n_trials=self.trials)

            print("Number of finished trials: {}".format(len(study_rf.trials)))

            print("Best trial:")
            trial = study_rf.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            return study_rf

        ### output-vif dataset ###
        Best is trial 17 with value: 1.2025391441833588.
        Number of finished trials: 20
        Best trial:
        Value: 1.2025391441833588
        Params:
            n_estimators: 237
            max_depth: 8
            min_samples_leaf: 9
            min_samples_split: 4
            max_features: sqrt
        
        ### output-all dataset ###
        Best is trial 19 with value: 1.193263311620101.
        Number of finished trials: 20
        Best trial:
        Value: 1.193263311620101
        Params:
            n_estimators: 133
            max_depth: 8
            min_samples_leaf: 7
            min_samples_split: 6
            max_features: sqrt
        ```

        ```
        2. LightGBM
        class LightGBM():
        def __init__(self, X_train, y_train,trials):
            super(LightGBM,self).__init__()
            self.X = X_train
            self.y = y_train
            self.trials = trials
        
        def lgb_objective(self,trial):
            lgb_learning_rate = trial.suggest_float('learning_rate', 0.04, 0.4)
            lgb_leaves= trial.suggest_int('num_leaves', 10, 1000)
            lgb_bytree = trial.suggest_float("colsample_bytree", 0.1,0.3)
            lgb_subsample = trial.suggest_float("subsample", 0.1,0.3)
            lgb_depth =  trial.suggest_int('max_depth', 3, 100)
            lgb_child_samples = trial.suggest_int('min_child_samples', 3, 2000)
            lgb_alpha =  trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
            # lgb_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
            lgb_smooth = trial.suggest_int('cat_smooth', 1, 100)
            lgb_gain_to_split = trial.suggest_float('min_split_gain', 0.0, 30.0)
            lgb_max_bin = trial.suggest_int('max_bin',2,100)
            lgb_boosting = trial.suggest_categorical('boosting_type', ['gbdt','dart'])
            # lgb_bagging = trial.suggest_uniform('bagging_fraction', 0.1, 1.0)
            # lgb_bagging_freq =  trial.suggest_int('bagging_freq', 0, 15)

            regressor_obj = LGBMRegressor(boosting_type=lgb_boosting,objective='regression', metric='rmse', verbosity = -1,num_leaves=lgb_leaves,learning_rate=lgb_learning_rate
                                        ,colsample_bytree= lgb_bytree, subsample=lgb_subsample, max_depth=lgb_depth, min_child_samples=lgb_child_samples,reg_alpha=lgb_alpha
                                        ,cat_smooth=lgb_smooth,min_split_gain=lgb_gain_to_split,max_bin = lgb_max_bin)
                                    

            rmse = np.sqrt(-cross_val_score(regressor_obj, self.X, self.y, scoring="neg_mean_squared_error", cv = 12, n_jobs=8))
            rmse =  np.mean(rmse)
            return rmse

        def tuning(self):
            
            sampler = TPESampler(seed=42) # TPESampler

            study_lgb = optuna.create_study(direction='minimize', sampler=sampler)
            study_lgb.optimize(self.lgb_objective, n_trials=self.trials)

            print("Number of finished trials: {}".format(len(study_lgb.trials)))

            print("Best trial:")
            trial = study_lgb.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            
            return study_lgb

        ### output-vif dataset ###
        Best is trial 5 with value: 1.2553408994889568.
        Number of finished trials: 20
        Best trial:
        Value: 1.2553408994889568
        Params:
            learning_rate: 0.17992382428821355
            num_leaves: 278
            colsample_bytree: 0.26574750183038587
            subsample: 0.17135066533871784
            max_depth: 30
            min_child_samples: 1087
            reg_alpha: 1.8548894229694903e-07
            cat_smooth: 81
            min_split_gain: 2.236519310393125
            max_bin: 99
            boosting_type: gbdt
        
        ### output-all dataset ###
        Best is trial 5 with value: 1.2513594445179432.
        Number of finished trials: 20
        Best trial:
        Value: 1.2513594445179432
        Params:
            learning_rate: 0.17992382428821355
            num_leaves: 278
            colsample_bytree: 0.26574750183038587
            subsample: 0.17135066533871784
            max_depth: 30
            min_child_samples: 1087
            reg_alpha: 1.8548894229694903e-07
            cat_smooth: 81
            min_split_gain: 2.236519310393125
            max_bin: 99
            boosting_type: gbdt
        ```

        ```
        3. XGBoost
        class XGBoost():
        def __init__(self, X_train, y_train,trials):
            warnings.filterwarnings('ignore')
            super(XGBoost,self).__init__()
            self.X = X_train
            self.y = y_train
            self.trials = trials
        
        def xgb_objective(self,trial):
            xgb_estimators = trial.suggest_int('n_estimators', 100, 300)
            xgb_depth = trial.suggest_int('max_depth', 3,10)
            xgb_learning_rate = trial.suggest_uniform('learning_rate', 0.01, 1)
            xgb_alpha = trial.suggest_uniform('reg_alpha',0.,1)
            # xgb_lamda = trial.suggest_uniform('reg_lambda',0.,1)

            regressor_obj = XGBRegressor(n_estimators=xgb_estimators,max_depth=xgb_depth,
                                                learning_rate = xgb_learning_rate,
                                                reg_alpha=xgb_alpha)

            rmse = np.sqrt(-cross_val_score(regressor_obj, self.X, self.y, scoring="neg_mean_squared_error", cv = 10, n_jobs=8))
            rmse = rmse.min()   
            return rmse
        
        def tuning(self):
            sampler = TPESampler(seed=42) # TPESampler

            study_xgb = optuna.create_study(direction='minimize', sampler=sampler)
            study_xgb.optimize(self.xgb_objective, n_trials=self.trials)

            print("Number of finished trials: {}".format(len(study_xgb.trials)))

            print("Best trial:")
            trial = study_xgb.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            return study_xgb
        
        ### output-vif dataset ###
        Best is trial 11 with value: 1.1943475595930018.
        Number of finished trials: 20
        Best trial:
        Value: 1.1943475595930018
        Params:
            n_estimators: 266
            max_depth: 7
            learning_rate: 0.0249927947317293
            reg_alpha: 0.9798341964022114

        ### output-all dataset ###
        Best is trial 1 with value: 1.1932953854658148.
        Number of finished trials: 20
        Best trial:
        Value: 1.1932953854658148
        Params:
            n_estimators: 131
            max_depth: 4
            learning_rate: 0.06750277604651747
            reg_alpha: 0.8661761457749352
        ```

        ```
        4. Stacking
        class Stacking():
        def __init__(self, X_train, y_train,model_list):
            self.model_list = model_list
            self.X = X_train
            self.y = y_train
        
            alpha = 0.0033000000000000004
            self.lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                                    alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                                    alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                                    alpha * 1.4], 
                            max_iter = 50000, cv = 10, n_jobs=-1)
            self.lasso = make_pipeline(RobustScaler(),self.lasso)

            alpha = 0.6
            self.ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                                    alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                                    alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], cv = 10)
            self.ridge = make_pipeline(RobustScaler(),self.ridge)

        def stack(self):
            rf_best = self.model_list[0].best_params
            lgb_best = self.model_list[1].best_params
            xgb_best = self.model_list[2].best_params
            del(lgb_best['learning_rate'])

            stack = StackingCVRegressor(regressors=(LGBMRegressor(**lgb_best),XGBRegressor(**xgb_best),self.lasso,self.ridge),
                                            meta_regressor=LGBMRegressor(**lgb_best),
                                            use_features_in_secondary=True,
                                            n_jobs=-1,cv=10)

            stack_model = stack.fit(self.X, self.y)

            return stack_model
        ```

    3. 검증 및 평가 
        - 학습이 끝난 XGBoost, LGBM, Random Forest, Stacking 예측 모델에 검증 데이터를 이용하여 예측을 진행하고 예측값과 실제값 간에 성능 평가를 진행한다.
        - 평가를 위해 MSE, MAE, RMSE를 사용한다. 

        ```
        class Tester():

        def __init__(self,X_train,y_train,X_test,y_test):
            
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.mode = ['lgb','xgb','rf']
        
        def test(self,model_list):
            pred = []
            for model,mode in zip(model_list,self.mode):
                if mode == 'lgb' :
                    model = LGBMRegressor(**model.best_params)
                    model.fit(self.X_train,self.y_train)
                    pred.append(model.predict(self.X_test))

                elif mode == 'xgb' :
                    model = XGBRegressor(**model.best_params)
                    model.fit(self.X_train,self.y_train)
                    pred.append(model.predict(self.X_test))
                else:
                    model = RandomForestRegressor(**model.best_params)
                    model.fit(self.X_train,self.y_train)
                    pred.append(model.predict(self.X_test))
            return pred
                
        def compute_score(self,pred_list):
            print(f'MSE --> random forest : {mean_squared_error(self.y_test,pred_list[0][2])}, LightGBM : {mean_squared_error(self.y_test,pred_list[0][0])},XGBoost : {mean_squared_error(self.y_test,pred_list[0][1])}, Stacking : {mean_squared_error(self.y_test,pred_list[1])}')

            print(f'MAE --> random forest : {mean_absolute_error(self.y_test,pred_list[0][2])}, LightGBM : {mean_absolute_error(self.y_test,pred_list[0][0])}, XGBoost : {mean_absolute_error(self.y_test,pred_list[0][1])}, Stacking : {mean_absolute_error(self.y_test,pred_list[1])}')

            print(f'R2 --> random forest : {r2_score(self.y_test,pred_list[0][2])}, LightGBM : {r2_score(self.y_test,pred_list[0][0])} , XGBoost : {r2_score(self.y_test,pred_list[0][1])}, Stacking : {r2_score(self.y_test,pred_list[1])}')

        ### output-vif dataset ###
        MSE --> random forest : 1.6165201520318637, LightGBM : 1.5865694422020291,XGBoost : 1.6046171868253996, Stacking : 1.6290328437614618
        MAE --> random forest : 0.9334316870379487, LightGBM : 0.9131079421306004, XGBoost : 0.9129578504899342, Stacking : 0.9181181113270899
        R2 --> random forest : 0.437051462039188, LightGBM : 0.44748171141677917 , XGBoost : 0.44119663576433765, Stacking : 0.43269395278928613

        ### output-all dataset ###
        MSE --> random forest : 1.5890621223765875, LightGBM : 1.5839747259912147,XGBoost : 1.5968371902545766, Stacking : 1.5961991952322516
        MAE --> random forest : 0.9087163627366381, LightGBM : 0.9110447577662901, XGBoost : 0.9108789670616239, Stacking : 0.9112111735556487
        R2 --> random forest : 0.4466136426468924, LightGBM : 0.44838531394562187 , XGBoost : 0.44390599740723446, Stacking : 0.4441281773563014
        ```

        - **최종 결과**

        | Model | MSE(vif/all) | MAE(vif/all) | R2(vif/all) |
        |-------|--------------|--------------|-------------|
        |XGBoost|  1.604/1.597 | 0.913/0.911  | 0.441/0.444 |
        | LGBM  |  1.587/1.584 | 0.913/0.911  | 0.447/0.448 |
        |  RF   |  1.617/1.589 | 0.933/0.909  | 0.437/0.447 | 
        |Stacking| 1.629/1.596 | 0.918/0.911  | 0.433/0.444 |

    4. 사후 분석
        - 의료 도메인에 적용되는 인공지능의 경우, 환자의 생명과 연관되기 때문에 모델이 내린 예측의 신뢰성과 해당 예측에 대한 타당성 확보가 중요하다. 따라서 SHAP을 적용하여 중요 변수를 추출해 예측의 타당성 및 모델의 신뢰성을 확인하고자 한다. 
        - 3단계에서 검증 및 평가를 통해 구한 결과를 보면 vif-dataset에 비해 모든 변수를 사용한 데이터 셋에서 모든 평가 지표 상 더 좋은 결과(미묘하게)를 도출하는 것을 확인할 수 있었다. 
        - 이를 통해 제거한 변수를 모두 활용하는 것이 심뇌혈관 질환 발생빈도 예측에 더 유의미하다는 것을 확인할 수 있었다.
        - 그 중에서도 vif,all dataset에서 4개의 예측 모델 중 LGBM이 가장 좋은 성능을 보이는 것을 확인했다.
        - 따라서, SHAP을 활용해 vif & all dataset 의 변수 중 높은 중요도를 갖는 변수를 확인해보고자 한다.
        - vif, all dataset 에서 높은 중요도를 보이는 것은 인구수와 지역 변수이다. 이는 둘 다 인구와 관련된 변수들로 크게 두 가지 관점으로 바라보고 있다.
        - 먼저 연령대의 차이가 혈관질환에 영향을 미친것으로 보인다. 심뇌혈관의 경우 청년층보다 중장년층에서 더 빈번하게 일어난다고 한다. 사후 분석 결과 역시 청년층보다 장년층과 관련된 변수를 더 우위에 둠으로써 이를 뒷받침하고 있다.
        - 두 번째로 지역 변수 역시 인구수의 영향이 많은 것으로 보인다. 지역의 경우, 다른 자연현상과도 관련이 있겠지만 우리나라의 경우 서울, 부산 등 일부 특정 지역에 인구가 과도하게 몰려있는 특성을 가진다. 다른 상황이 모두 같다고 가정할 때, 사람이 많은 지역에서 심뇌혈관질환의 발생빈도는 사람이 적은 지역보다 빈번할 수밖에 없다. 
        - 이외에 인상적인 변수는 미세먼지 농도 변수이다. 심뇌혈관의 경우, 미세먼지가 심해질수록 심뇌혈관질환 발생 가능성이 증가하는 것으로 알려져 있다. 사후분석 결과를 통해 모델이 사전 정보 없이도 이를 제대로 파악하고 있음을 확인하였다. 
        - 예보 데이터의 경우, 풍향, 강수확률, 일 최고기온이 중요 변수로 산출되었는데 이 중 특히 유의미한 변수로는 일 최고기온으로, 혈관은 추울 때도, 더울 때도 기온의 영향을 많이 받기 때문이라고 생각하고 있다. 
        - 또한 풍향 역시 기온과 같은 맥락의 변수로 추측되는데, 우리나라의 경우 4계절에 따른 풍향의 방향이 다르므로 해당 변수가 계절과 관련된 정보를 담고 있는 것으로 추측되어진다. 


        ```
        # Shap
        plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.facecolor'] = 'white'
        shap_values = shap.TreeExplainer(model).shap_values(self.X_train)
        shap.summary_plot(shap_values, self.X_train, plot_type="bar", max_display=int(len(self.X_train.columns)))
        plt.savefig(f'./Emsemble/img/{self.mode}_{len(self.X_train.columns)}.png')
        ```
        
        - LGBM (vif dataset)
        <p align='center'><img src="./img/vif_lgbm.png" width='600' height='600'></p>

        - LGBM (all dataset)
        <p align='center'><img src="./img/all_lgbm.png" width='600' height='600'></p>
    5. 결론

    - 이번 튜토리얼에서 지역 변수 및 인구수 등과 같은 여러 독립변수를 활용하여 심뇌혈관 질환 발생 빈도 예측을 위해 여러 예측 모델을 사용하였다. 
    - 변수가 많다 보니 vif 방법을 통해 변수를 제거한 데이터와 전체 변수를 사용한 데이터와의 비교도 진행하여 제거된 변수가 심뇌혈관 질환 발생 빈도수에 어느 정도 영향을 끼치는 것을 확인할 수 있었다. 
    - 여러 모델에 대한 최적의 파라미터를 찾기 위해 optuna를 활용한 하이퍼 파라미터 튜닝도 진행하여 최적의 파라미터 탐색도 진행해보았다. 
    - 여러 예측 모델 중 가장 좋은 모델로는 lgbm으로 해당 모델에 대한 사후 분석 결과도 확인해보았다.
    - 튜토리얼을 진행하면서 인생 깊었던 부분은 실제 이론상으로 lgbm이 다른 모델에 비해 메모리 사용량도 적고 학습 시간도 적다고 배웠는데, 직접 모델을 구현하고 돌리면서 다른 모델에 비해 학습 속도가 현저하게 빠른 것을 확인하였다. 특히 optuna를 활용한 하이퍼 파라미터 튜닝을 진행할 때 xgboost, random forest에 비해 훨씬 더 빨리 파라미터 튜닝을 진행하는 것도 확인할 수 있었다.

    
    

**Reference**
- swallow.github.io
- https://julie-tech.tistory.com/119
- https://tyami.github.io/machine%20learning/ensemble-1-basics/
- https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-11-%EC%95%99%EC%83%81%EB%B8%94-%ED%95%99%EC%8A%B5-Ensemble-Learning-%EB%B0%B0%EA%B9%85Bagging%EA%B3%BC-%EB%B6%80%EC%8A%A4%ED%8C%85Boosting
- https://assaeunji.github.io/machine%20learning/2020-09-05-gbm/
- https://bioinformaticsandme.tistory.com/167