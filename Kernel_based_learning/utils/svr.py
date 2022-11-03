import numpy as np
import optuna
import sklearn.datasets as d
import pandas as pd
import sklearn
import sklearn.svm as svm
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions 
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

class SVR:
    def __init__(self,args,dataset):
        self.args = args
        self.dataset = dataset

    def regression(self):
        X,y = self.dataset.data, self.dataset.target 
        X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(X,y,random_state=0)
        
        if self.args.kernel =='linear':
            svm_reg = svm.SVR(kernel=self.args.kernel,C=1.0,epsilon=0.1)
            svm_reg.fit(X_train,y_train)
        else :
            svm_reg = svm.SVR(kernel=self.args.kernel,C=1.0,epsilon=0.1,degree=3,coef0=0.1)
            svm_reg.fit(X_train,y_train)

        y_pred = svm_reg.predict(X_val)
        self.metric(y_val,y_pred)
 
    
    def gridSearch(self):
        X,y = self.dataset.data, self.dataset.target
        X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(X,y,random_state=0)

        if self.args.kernel == 'linear':
            svm_reg = svm.SVR(kernel='linear')
            svr_parameters = {'C':[0.001,0.01,0.1,1,10,25,50,100],'epsilon':[0.1,0.3,0.5,0.8]}
            grid_svr = GridSearchCV(svm_reg,param_grid=svr_parameters,cv=5,scoring='r2')
            grid_svr.fit(X_train,y_train)
        else:
            svm_reg = svm.SVR(kernel='poly')
            svr_non_parameters = {'C':[0.001,0.01,0.1,1,10,25,50,100],'epsilon':[0.1,0.3,0.5,0.8],'degree':[3,4,5],'coef0':[0.0,0.1,0.2,0.3]}
            grid_svr = GridSearchCV(svm_reg,param_grid=svr_non_parameters,cv=5,scoring='r2')
            grid_svr.fit(X_train,y_train)

        print('optimal parameter:',grid_svr.best_params_)
        print('optimal value:',grid_svr.best_score_)

        if self.args.kernel=='linear':
            model_best_params = grid_svr.best_params_
            model_best_params['kernel'] = self.args.kernel
            model_best_params['gamma'] = 'scale'
            model = svm.SVR(**model_best_params)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            self.metric(y_val, y_pred)
        else:
            model_best_params = grid_svr.best_params_
            model_best_params['kernel'] = self.args.kernel
            model = svm.SVR(**model_best_params)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            self.metric(y_val, y_pred)

        result = pd.DataFrame(grid_svr.cv_results_['params'])
        result['mean_test_score'] = grid_svr.cv_results_['mean_test_score']
        result.sort_values(by='mean_test_score',ascending=False)

        return result,grid_svr.best_params_
    
    def showplt(self,best_params):
        X,y = self.dataset.data, self.dataset.target
        pca = PCA(n_components=1)
        pca_X = pca.fit_transform(X)

        ## SVR - linear
        if self.args.kernel=='linear':
            model_best_params = best_params
            model_best_params['kernel'] = self.args.kernel
            model_best_params['gamma'] = 'scale'
            model = svm.SVR(**model_best_params)
        else:
            model_best_params = best_params
            model_best_params['kernel'] = self.args.kernel
            model = svm.SVR(**model_best_params)

        model.fit(pca_X,y)
        y_pred = model.predict(pca_X)
        self.metric(y, y_pred)

        plt.title(self.args.kernel)
        plt.scatter(pca_X,y)
        plt.scatter(pca_X,y_pred,color='r')
        plt.show()

    def metric(self,y_true,y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')
