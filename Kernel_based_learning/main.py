import os
from args import get_args
from svc import SVC
from svr import SVR
import sklearn.datasets as d

def main():
    args = get_args()

    if args.model == 'svc':
        dataset = d.load_breast_cancer()
        model = SVC(args,dataset)
        model.cross_validation()
        best_params = model.fine_tuning()
        model.showplt2(best_params)
    else: 
        dataset = d.load_boston()
        model = SVR(args,dataset)
        model.regression()
        result, best_params = model.gridSearch()
        print('optimal hyperparameter',result)
        model.showplt(best_params)

    

if __name__ == '__main__':
    main()