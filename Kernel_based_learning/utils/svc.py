import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn.svm as svm 
import sklearn.metrics as mt 
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.decomposition import PCA
import optuna
import warnings
import sklearn 
from mlxtend.plotting import plot_decision_regions 

warnings.filterwarnings('ignore')

class SVC:
    def __init__(self,args,dataset):
        self.args = args 
        self.dataset = dataset

    def cross_validation(self):
        X,y = self.dataset.data, self.dataset.target
        svm_clf = svm.SVC(kernel=self.args.kernel)
        print(pd.DataFrame(cross_validate(svm_clf,X,y,cv=6)))
        print('Cross-validation result average:', cross_val_score(svm_clf,X,y,cv=6).mean())
    
    def parameter_tuning(self):
        def objective(trial):

            if self.args.kernel == 'linear':
                svc_c = trial.suggest_float('C',1e-4,1e2,log=True)
                classifier_obj = svm.SVC(C=svc_c, gamma='auto',kernel=self.args.kernel)
            else : 
                svc_c = trial.suggest_float('C',1e-4,1e-2,log=True)
                svc_degree = trial.suggest_int('degree',3,5,step=1)
                svc_gamma = trial.suggest_categorical('gamma',['scale','auto'])
                svc_coef = trial.suggest_float('coef0',0.0,0.3,step=0.1)
                classifier_obj = svm.SVC(C=svc_c,degree=svc_degree,gamma=svc_gamma,coef0=svc_coef,kernel=self.args.kernel)

            X,y = self.dataset.data,self.dataset.target 
            X_train,X_val,y_train,y_val = sklearn.model_selection.train_test_split(X,y,random_state=0)
            
            classifier_obj.fit(X_train,y_train)
            y_pred = classifier_obj.predict(X_val)

            accuracy = sklearn.metrics.accuracy_score(y_val,y_pred,normalize=True)
            return accuracy
            
        warnings.filterwarnings('ignore')
        study = optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=100)

        print(f'-----------{self.args.kernel}-----------')
        print('Optimal hyperparameter among 100 trials:',study.best_trial.params)
        print('Highest accuracy among 100 trials:',study.best_trial.value)

        return study.best_params
    
    ## plot function 
    def showplt(self,x,y,svm1):
        
        title = f'{self.args.kernel}_SVC'
        plt.scatter(x[:,0],x[:,1], c=y, s=30, cmap=plt.cm.Paired)

        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = svm1.decision_function(xy).reshape(XX.shape)

        #margins,decision boundary
        ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-', '--'])

        #Support Vector
        ax.scatter(svm1.support_vectors_[:,0], svm1.support_vectors_[:,1], s=60, facecolors='r')
        plt.title(title)
        plt.show()

    def showplt2(self,best_params):
        X,y = self.dataset.data, self.dataset.target
        pca = PCA(n_components=2)
        pca_X = pca.fit_transform(X)

        if self.args.kernel == 'linear':
            model_best_params = best_params
            model_best_params['kernel'] = 'linear'
            model_best_params['gamma'] = 'auto'
            model = svm.SVC(**model_best_params)
        else :
            model_best_params = best_params
            model_best_params['kernel'] = 'rbf'
            model = svm.SVC(**model_best_params)

        model.fit(pca_X,y)

        plot_decision_regions(X=pca_X,y=y, clf=model, legend=2)
        plt.show()
    
    def prepare_plt(self,best_params):
        X,y = self.dataset.data, self.dataset.target
        pca = PCA(n_components=2)
        pca_X = pca.fit_transform(X)

        if self.args.kernel == 'linear':
            model_best_params = best_params
            model_best_params['kernel'] = 'linear'
            model_best_params['gamma'] = 'auto'
            model = svm.SVC(**model_best_params)
        else :
            model_best_params = best_params
            model_best_params['kernel'] = 'rbf'
            model = svm.SVC(**model_best_params)

        model.fit(pca_X,y)

        return model, pca_X, y 

    