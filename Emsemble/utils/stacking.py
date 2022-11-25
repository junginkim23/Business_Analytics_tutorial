from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings



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