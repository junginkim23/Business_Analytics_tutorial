## Dimensionality Reduction 



1. 차원 축소란?
2. 지도 학습 기반 차원 축소 방법 
    - Forward selection
    - Backward elimination 
    - Stepwise selection
    - Genetic algorithm 
3. 비지도학습 차원 축소 방법 
    - Principal component analysis (PCA)
    - Multi-dimensional scaling (MDS)
    - ISOMAP, LLE, t-SNE 

---

### 1.차원 축소란? 

차원 축소는 매우 많은 피처로 구성되어 있는 다차원 데이터 집합의 차원을 축소하여 새로운 차원의 데이터 집합을 구성하는 것을 말한다. 선택된 데이터 집합은 성능을 저하시키지 않는 선에서 선택되며 이를 통해 변수간 독립성을 확보하게 된다. 이러한 차원 축소 기법에는 크게 변수 선택, 변수 추출 기법이 존재한다.

![Feature_extraction&selection](./image/dimensionality_reduction.png)
