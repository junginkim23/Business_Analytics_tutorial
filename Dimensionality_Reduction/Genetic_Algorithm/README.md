## ๐ Python Tutorial - Genetic Algorithm


`Procedure`

- Step 1. Initiation 
- Step 2. Training model
- Step 3. Fitness evaluation 
- Step 4. Selection
- Step 5. Crossover & Mutation
- Step 6. Choosing the final set of variables

---


### Step 1. Initiation 
- ์ง์ ํ chromosome(=population_size) ์์ ํด๋น chromosome์ ๋ณ์ ๊ฐ์๋งํผ ์ด์ง ๊ฐ ์์ฑ
```
def __initialize(self, n_genes: int):
    # Create population_size chromosomes
    self.population = np.random.randint(
        2, size=(self.population_size, n_genes))

==> [output] (chromosome์ ์: 10, Gene(๋ณ์)์ ์: 13)
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
 - 1 ๋จ๊ณ์์ ์์ฑํ ๊ฐ chromosome์ ์ด์ง ๊ฐ์ ๋ฐ๋ผ ์ ํ๋ ๋ณ์๋ง์ ์ฌ์ฉํด ๋ชจ๋ธ ํ์ต์ ์งํํ๊ณ  ๋ถ๋ฅ ์ฑ๋ฅ ์งํ๋ก accuracy๋ฅผ ์ฌ์ฉํด ๊ฐ chromosome ๋ง๋ค ์ ํฉ๋ ํ๊ฐ๋ฅผ ์งํํ๋ค. 
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
    #ย Pool for parallelization
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
- ์ฌ๋ฌ chromosome ์ค ์ ์ ์ ์กฐํฉ์ ์งํํ  chromosome์ ์ ํํ๋ ๋จ๊ณ. ์ ํ ๊ธฐ๋ฒ์ผ๋ก๋ Roulette method, Ranking, Steady state selection, Tournament-based selection, Elitism์ด ์กด์ฌํ๋ค. ๊ทธ ์ค์์ Tournament-based selection ๊ธฐ๋ฒ์ ์ฌ์ฉ. ํด๋น ๊ธฐ๋ฒ์ ํ ๋๋จผํธ๋ฅผ ์งํํ  ๊ฐ์ฒด ์๋งํผ ๋๋คํ ๋์๋ฅผ ๋ฝ๊ณ  ์ ํ๋ ๋์(์ ์)์ ํด๋นํ๋ chromosome์ ์ฑ๋ฅ์ด ๋ ์ข์ chromosome์ ์ ํํ๊ฒ ๋๋ค. 

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
- Crossover: 4๋จ๊ณ์์ ์ ํ๋ ๋ถ๋ชจ ์ผ์์ฒด๋ค์ ์ ์ ์ ์ ๋ณด๋ฅผ ์๋ก ๊ตํํ์ฌ ์๋ก์ด ์์ ์ผ์์ฒด๋ฅผ ์์ฑํ๋ ๋จ๊ณ์ด๋ค. ์ฝ๋ ์์์๋ ๊ต๋ฐฐ์จ์ ์ง์ ํ์ฌ ๊ต๋ฐฐํ๋ ๋ถ๋ชจ ์ผ์์ฒด์ ์๋ฅผ ์ง์ ํ๋ค. ๊ทธ๋ฆฌ๊ณ  ๊ต๋ฐฐ๊ฐ ๋์ง ์๋ ์ผ์์ฒด์ ๊ฒฝ์ฐ ๊ทธ๋๋ก ์์ ์ผ์์ฒด๋ก ์ฌ์ฉํ๊ฒ ๋๋ค. 
- Mutation: ์ผ๋ถ ์ ์ ์๋ฅผ ๋ณํ์์ผ ์๋ก์ด ์ ์ ์ ์กฐํฉ์ ์์ฑํ๋ ๋จ๊ณ์ด๋ค. ํด๋น ๋จ๊ณ๋ฅผ ํตํด Local optimum์ ๋น ์ง ์ ์๋ ์ํ์ ์ด๋ ์ ๋ ์ ๊ฑฐํ์ฌ global optimum์ผ๋ก ์๋ ดํ  ์ ์๊ฒ ํ๋ค. ๋ณธ ์ฝ๋์์๋ Mutation ๋ฐ๋ณต ์๋ฅผ ์ฌ์ ์ ์ง์ ํ ๋์ฐ๋ณ์ด์จ์ ํตํด ๊ณ์ฐํ๊ณ  ํด๋น ๋ฐ๋ณต ์๋งํผ ๋๋ค์ผ๋ก ์ผ์์ฒด์ ๋ณ์๋ฅผ ์ ํํด ํด๋น ์ผ์์ฒด์ ๋ณ์๊ฐ 0์ด๋ฉด 1๋ก 1์ด๋ฉด 0์ผ๋ก ๋ณํ์์ผฐ๋ค.

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
- ์ผ๋ จ์ ๊ณผ์ ์ ํตํด ๊ฐ์ฅ ๋์ ์ ํฉ๋ ํจ์ ๊ฐ์ ๊ฐ๋ ์ผ์์ฒด์ ์ธ์ฝ๋ฉ ๋ ๋ณ์ ์กฐํฉ์ ์ ํํ๋ ๋จ๊ณ์ด๋ค. ๋ณธ ์ฝ๋์์๋ ์ผ๋ จ์ ๊ณผ์ ์ ๊ฑฐ์ณ ๋ง๋ค์ด์ง ์์ ์ผ์์ฒด๋ค์ ๋ณ์ ์กฐํฉ์ผ๋ก ๋ถ๋ฅ ๋ชจ๋ธ์ ์ฑ๋ฅ์ ํ๊ฐํ๊ณ  ๊ทธ ์ค์์ ๊ฐ์ฅ ์ข์ ๋ณ์ ์กฐํฉ์ ์ฐพ์๋ธ๋ค.

```
==> [output]
# (BETTER) A better chromosome than the current one has been found (0.9788177339901478).
# Current best chromosome: (array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]), 0.9788177339901478, 1)
```

**reference: https://github.com/BiDAlab/GeneticAlgorithm**