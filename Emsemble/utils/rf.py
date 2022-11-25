import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from optuna.samplers import TPESampler
import optuna
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.reset_option('all')

warnings.filterwarnings(action='ignore') 

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
        sampler = TPESampler(seed=42) # TPESampler --> MAE가 최소가 되는 방향으로 학습 진행 (MAE: 평균절대오차!)

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
