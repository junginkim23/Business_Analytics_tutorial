from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

dataset = datasets.load_diabetes()

X = pd.DataFrame(data=dataset['data'])
y = pd.Series(data=dataset['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1225)

model = sm.OLS(y_train,X_train).fit()

def forward_selection(X_train,y_train):
   
    variables = X_train.columns.tolist() 
    y = y_train 
  
    forward_variables = []
    
    sl_enter = 0.05
    sl_remove = 0.05

    
    sv_per_step = [] 
  
    adj_r_squared_list = []
    
    steps = []
    step = 0


    while len(variables) > 0:
        remainder = list(set(variables) - set(forward_variables))
        pval = pd.Series(index=remainder)
        
        for col in remainder: 
            X = X_train[forward_variables+[col]]
            X = sm.add_constant(X)
            model = sm.OLS(y,X).fit(disp=0)
            pval[col] = model.pvalues[col]
    
        min_pval = pval.min()
        if min_pval < sl_enter: 
            forward_variables.append(pval.idxmin())
          
            while len(forward_variables) > 0:
                selected_X = X_train[forward_variables]
                selected_X = sm.add_constant(selected_X)
                selected_pval = sm.OLS(y,selected_X).fit(disp=0).pvalues[1:] 
                max_pval = selected_pval.max()
                if max_pval >= sl_remove: 
                    remove_variable = selected_pval.idxmax()
                    forward_variables.remove(remove_variable)
                else:
                    break
            
            step += 1
            steps.append(step)
            adj_r_squared = sm.OLS(y,sm.add_constant(X_train[forward_variables])).fit(disp=0).rsquared_adj
            adj_r_squared_list.append(adj_r_squared)
            sv_per_step.append(forward_variables.copy())
        else:
            break
    return forward_variables, steps, adj_r_squared, sv_per_step