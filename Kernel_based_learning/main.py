import os
from utils.args import get_args
from utils.svc import SVC
from utils.svr import SVR
import sklearn.datasets as d

def main():
    args = get_args()

    if args.model == 'svc':
        dataset = d.load_breast_cancer()
        model = SVC(args,dataset)
        model.cross_validation()
        best_params = model.parameter_tuning()
        p_model,X,y = model.prepare_plt(best_params)
        model.showplt(X,y,p_model)
        model.showplt3(dataset.data, dataset.target)
    else: 
        dataset = d.load_boston()
        model = SVR(args,dataset)
        model.regression()
        result, best_params = model.gridSearch()
        print('optimal hyperparameter',result)
        model.showplt(best_params)

    

if __name__ == '__main__':
    main()