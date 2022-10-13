## Python Tutorial - Genetic Algorithm


`Procedure`

- Step 1. Initiation 
- Step 2. Training model
- Step 3. Fitness evaluation 
- Step 4. Selection
- Step 5. Crossover & Mutation
- Step 6. Choosing the final set of variables

---


### Step 1. Initiation 
- 지정한 chromosome(=population_size) 수와 해당 chromosome에 변수 개수만큼 이진 값 생성
```
def __initialize(self, n_genes: int):
    # Create population_size chromosomes
    self.population = np.random.randint(
        2, size=(self.population_size, n_genes))

==> [output] (chromosome의 수: 10, Gene(변수)의 수: 13)
[[0 1 0 0 0 1 0 0 0 1 0 0 0]
 [0 1 0 1 1 1 0 1 0 1 1 1 1]
 [1 1 1 1 0 0 1 1 1 0 1 0 0]
 [0 0 0 1 1 1 1 1 0 1 1 0 1]
 [0 1 0 1 1 0 0 0 0 0 0 0 0]
 [1 1 0 1 1 1 1 0 1 0 1 1 1]
 [0 1 0 1 0 1 0 0 1 0 1 1 1]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]
 [1 1 1 1 1 0 1 0 1 1 0 1 0]
 [1 1 0 1 0 1 0 0 1 1 0 1 1]]
 ```

 ### Step 2&3. Training model/Fitness evaluation 
 - 1 단계에서 생성한 각 chromosome의 이진 값에 따라 선택된 변수만을 사용해 모델 학습을 진행하고 분류 성능 지표로 accuracy를 사용해 각 chromosome 마다 적합도 평가를 진행한다. 
```
def estimate(self, chromosome: np.ndarray) -> float:
    # Select those features with ones in chromosome
    data = self.X.loc[:, chromosome.astype(bool)]

    # Cross-validation execution
    scores = cross_validate(
        self.estimator, data, self.y, cv=self.cv, scoring=self.scorer,
        verbose=self.verbose, return_train_score=self.calc_train_score)

    if self.calc_train_score:
        return (scores['test_score'].mean(),
                scores['train_score'].mean())
    else:
        return (scores['test_score'].mean(), None)

def __evaluate(self, population: np.ndarray) -> np.ndarray:
    # Pool for parallelization
    pool = mp.Pool(self.n_jobs)
    return np.array(pool.map(self.estimate, population))

# Insert initial best chromosome if is defined
if self.initial_best_chromosome is None:

    # Evaluate initial population and update best_chromosome
    print(f'# Evaluating initial population...')
    population_scores = self.__evaluate(self.population)
    best_chromosome_index = np.argmax(population_scores[:, 0])

elif type(self.initial_best_chromosome) == np.ndarray and \
    len(self.initial_best_chromosome) == X.shape[1] and \
    len(np.where((self.initial_best_chromosome <= 1) &
                    (self.initial_best_chromosome >= 0))[0]):

    # Introduce the best in the population
    index_insertion = np.random.randint(self.population_size)
    self.population[index_insertion] = self.initial_best_chromosome

    # Evaluate initial population and update best_chromosome
    print(f'# Evaluating initial population...')
    population_scores = self.__evaluate(self.population)
    best_chromosome_index = index_insertion

==> [output]

# Evaluating initial population...
# Current best chromosome: (array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]), 0.9502463054187192, 0)
```

 ### Step 4. Selection
- 여러 chromosome 중 유전자 조합을 진행할 chromosome을 선택하는 단계. 선택 기법으로는 Roulette method, Ranking, Steady state selection, Tournament-based selection, Elitism이 존재한다. 그 중에서 Tournament-based selection 기법을 사용. 해당 기법은 토너먼트를 진행할 개체 수만큼 랜덤한 난수를 뽑고 선택된 난수(정수)에 해당하는 chromosome의 성능이 더 좋은 chromosome을 선택하게 된다. 

 ```
 def __selection(
        self, population: np.ndarray, population_scores: np.ndarray,
        best_chromosome_index: int) -> np.ndarray:
    # Create new population
    new_population = [population[best_chromosome_index]]

    # Tournament_k chromosome tournament until fill the numpy array
    while len(new_population) != self.population_size:
        # Generate tournament_k positions randomly
        k_chromosomes = np.random.randint(
            len(population), size=self.tournament_k)
        # Get the best one of these tournament_k chromosomes
        best_of_tournament_index = np.argmax(
            population_scores[k_chromosomes])
        # Append it to the new population
        new_population.append(
            population[k_chromosomes[best_of_tournament_index]])

    return np.array(new_population)

==> [output]
[[1 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]
 [1 1 1 1 0 0 1 1 1 0 1 0 0]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]
 [1 1 1 1 0 0 1 1 1 0 1 0 0]
 [1 1 0 1 0 1 0 0 1 1 0 1 1]
...
 [0 1 0 0 1 1 0 0 1 0 0 1 1]
 [1 1 0 1 1 1 1 0 1 0 0 1 1]
 [1 1 0 1 1 1 1 1 1 1 1 1 1]]
```

### Step 5. Crossover & Mutation
- Crossover: 4단계에서 선택된 부모 염색체들의 유전자 정보를 서로 교환하여 새로운 자식 염색체를 생성하는 단계이다. 코드 상에서는 교배율을 지정하여 교배하는 부모 염색체의 수를 지정한다. 그리고 교배가 되지 않는 염색체의 경우 그대로 자식 염색체로 사용하게 된다. 
- Mutation: 일부 유전자를 변형시켜 새로운 유전자 조합을 생성하는 단계이다. 해당 단계를 통해 Local optimum에 빠질 수 있는 위험을 어느 정도 제거하여 global optimum으로 수렴할 수 있게 한다. 본 코드에서는 Mutation 반복 수를 사전에 지정한 돌연변이율을 통해 계산하고 해당 반복 수만큼 랜덤으로 염색체와 변수를 선택해 해당 염색체의 변수가 0이면 1로 1이면 0으로 변환시켰다.

```
def __crossover(self, population: np.ndarray) -> np.ndarray:
    # Define the number of crosses
    n_crosses = int(self.crossover_rate * int(self.population_size / 2))

    # Make a copy from current population
    crossover_population = population.copy()

    # Make n_crosses crosses
    for i in range(0, n_crosses*2, 2):
        cut_index = np.random.randint(1, self.X.shape[1])
        tmp = crossover_population[i, cut_index:].copy()
        crossover_population[i, cut_index:], crossover_population[i+1,
                                                                    cut_index:] = crossover_population[i+1, cut_index:], tmp
        # Avoid null chromosomes
        if not all(crossover_population[i]):
            crossover_population[i] = population[i]
        if not all(crossover_population[i+1]):
            crossover_population[i+1] = population[i+1]

    return crossover_population

==> [output]
[[1 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]
 [1 1 1 1 0 0 1 1 1 0 1 0 0]
 [1 1 1 1 0 0 1 1 1 0 1 0 0]
 [1 1 0 1 0 1 0 0 1 1 0 1 1]
 [1 1 0 1 0 1 0 0 1 1 0 1 1]
 [1 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 1 1 1 1 1 1 0 0 1 1 1]]

 def __mutation(self, population: np.ndarray) -> np.ndarray:
    # Define number of mutations to do
    n_mutations = int(
        self.mutation_rate * self.population_size * self.X.shape[1])

    # Mutating n_mutations genes
    for _ in range(n_mutations):
        chromosome_index = np.random.randint(0, self.population_size)
        gene_index = np.random.randint(0, self.X.shape[1])
        population[chromosome_index, gene_index] = 0 if \
            population[chromosome_index, gene_index] == 1 else 1

    return population

==> [output]
[[1 1 0 1 0 0 0 0 0 0 1 1 0]
 [0 1 0 0 0 0 0 0 0 0 1 1 0]
 [1 1 0 0 1 0 1 0 0 0 1 0 0]
 [1 1 1 1 0 1 1 1 0 0 1 1 0]
 [1 1 1 1 1 1 1 1 0 1 1 1 1]
 [1 1 1 1 0 0 1 1 1 0 1 0 0]
 [1 1 1 1 1 1 0 1 0 0 0 1 1]
 [1 1 1 1 1 1 1 1 1 0 1 1 0]
 [1 1 0 1 0 1 0 0 1 0 0 1 1]
 [1 1 0 1 0 1 0 0 1 1 0 0 1]]
```

### Step 6. Choosing the final set of variables
- 일련의 과정을 통해 가장 높은 적합도 함수 값을 갖는 염색체에 인코딩 된 변수 조합을 선택하는 단계이다. 본 코드에서는 일련의 과정을 거쳐 만들어진 자식 염색체들의 변수 조합으로 분류 모델의 성능을 평가하고 그 중에서 가장 좋은 변수 조합을 찾아낸다.

```
==> [output]
# (BETTER) A better chromosome than the current one has been found (0.9788177339901478).
# Current best chromosome: (array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]), 0.9788177339901478, 1)
```

**reference: https://github.com/BiDAlab/GeneticAlgorithm**