from sklearn.preprocessing import LabelEncoder
import pandas as pd 
from sklearn.model_selection import train_test_split
from utils.vif import VIF

class MakeDataset():

    def __init__(self,args):

        self.args = args
        self.le = LabelEncoder()

    def make_data(self):
        # Simple preprocessing and standardization of train data

        data = pd.read_csv(self.args.data_path) # Replacing Missing values using k-means
        X_data = data.drop(['freq'],axis = 1)
        y_data = data['freq']

        # Used to encode stnNm features
        self.le.fit(X_data['stnNm'])
        X_data['stnNm'] = self.le.transform(X_data['stnNm'])

        # Converting to an integer type of feature related to the population 
        for col in X_data.columns:
            if X_data[col].dtype == 'object':
                X_data[col] = X_data[col].apply(lambda x : x.replace(',',''))
                X_data[col] = X_data[col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.2, random_state=328)

        if self.args.mode_1 == 'vif':
            X_train,X_test = VIF(X_train, X_test).after_vif()
            return X_train, X_test, y_train, y_test
        else:
            return X_train, X_test, y_train, y_test