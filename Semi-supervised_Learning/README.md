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

2. MixMatch
- 기존 준지도 학습 방법 Consistency Regularization, Entropy Minimization, Traditional Regularization (Mix Up)을 결합한 방법론 

3. FixMatch
- MixMatch와 ReMixMatch의 경우 성능 고도화를 위해 주요 기법들을 추가 및 혼합하는 방향으로 발전함 
- 지나치게 정교한 loss term과 조정하기 어려운 수 많은 사용자 정의 파라미터를 사용하는 형태

4. FlexMatch
- 분류가 쉬운 범주의 경우 처음부터 Confidence가 높은 데이터가 다수 Pseudo Labeling이 되어 계속 더 잘 학습할 수 있게 유도되지만 분류가 어려운 범주는 Confidence가 높은 레이블이 없는 데이터가 많지 않기 때문에 비지도 학습의 본 의도인 레이블이 없는 데이터의 정보 활용이 어렵다는 문제가 존재
- 따라서, 모든 클래스에 동일한 Confidence 기준을 적용하지 않고 각 클래스의 난이도에 따른 다른 기준을 적용하는 것이 핵심! 


---

**Tutorial** 