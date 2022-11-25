from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from optuna.samplers import TPESampler
import optuna
import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.reset_option('all')

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
        
        sampler = TPESampler(seed=42) # TPESampler --> MAE가 최소가 되는 방향으로 학습 진행 (MAE: 평균절대오차!)

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
        