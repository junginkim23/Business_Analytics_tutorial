from utils.dataset import MakeDataset
from utils.lgb import LightGBM
from utils.xgb import XGBoost
from utils.rf import RandomForest
from utils.stacking import Stacking
from utils.vif import VIF
from utils.test import Tester
from utils.args import get_args
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from collections import defaultdict
import warnings
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.reset_option('all')


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

    

if __name__ == '__main__':
    main()
