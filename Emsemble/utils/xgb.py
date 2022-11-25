from sklearn.model_selection import cross_val_score
import numpy as np
from optuna.samplers import TPESampler
import optuna
import pandas as pd 
from pandas import MultiIndex, Int16Dtype
from xgboost import XGBRegressor
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.reset_option('all')
 

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
        sampler = TPESampler(seed=42) # TPESampler --> MAE가 최소가 되는 방향으로 학습 진행 (MAE: 평균절대오차!)

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
