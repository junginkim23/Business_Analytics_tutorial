from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

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
                self.SHAP(model)
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
        print('-----------------------------------------------------------------------------------------------------')


    def SHAP(self,model):
        # Shap
        plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.facecolor'] = 'white'
        shap_values = shap.TreeExplainer(model).shap_values(self.X_train)
        shap.summary_plot(shap_values, self.X_train, plot_type="bar", max_display=int(len(self.X_train.columns)))
        plt.show()
        

    

