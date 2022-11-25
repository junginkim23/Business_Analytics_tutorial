from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd 
import time

class VIF():

    def __init__(self, X_train, X_test):
        self.X = X_train
        self.X_test = X_test
        self.remove_list = []
        self.t = time.time()

    def check_vif(self):
        dataframe = add_constant(self.X)
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(
            dataframe.values, i) for i in range(dataframe.shape[1])]
        vif["features"] = dataframe.columns
        return vif
    
    def after_vif(self):
        for _ in range(1000):
    
            vif = self.check_vif()

            if len(vif[vif['VIF Factor'] >=10]) >=1:
                vif_value = round(vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[0,0],3)
                remove_col = vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[0,1]

                if (remove_col == 'const') & (vif[vif['VIF Factor'] >=10].shape[0] == 1):
                    print('VIF가 10이 넘는 변수가 없습니다. === FOR LOOP을 종료합니다.')
                    break

                elif (remove_col == 'const') & (vif[vif['VIF Factor'] >=10].shape[0] != 1):
                    vif_value = round(vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[1,0],3)
                    remove_col = vif[vif['VIF Factor'] >=10].sort_values('VIF Factor', ascending=False).iloc[1,1]
                    self.remove_list.append(remove_col)
                    self.X = self.X.drop(remove_col, axis=1)
                    t2 = time.time()
                    elapsed_time = t2-self.t
                    print('VIF값이 '+str(vif_value)+'인 '+remove_col+'이 제거되었습니다. === 현재 총 제거된 변수의 개수는 '+str(len(self.remove_list))+'개 입니다. === 경과된 시간: '+str(round(elapsed_time/60))+'분')

                else:
                    self.remove_list.append(remove_col)
                    self.X = self.X.drop(remove_col, axis=1)
                    t2 = time.time()
                    elapsed_time = t2-self.t
                    print('VIF값이 '+str(vif_value)+'인 '+remove_col+'이 제거되었습니다. === 현재 총 제거된 변수의 개수는 '+str(len(self.remove_list))+'개 입니다. === 경과된 시간: '+str(round(elapsed_time/60))+'분')


            else:
                print('VIF가 10이 넘는 변수가 없습니다. === FOR LOOP을 종료합니다.')
                break

            
        vif_col = self.X.columns.to_list()
        test_vif = self.X_test[vif_col]

        print(vif_col)

        return self.X, test_vif