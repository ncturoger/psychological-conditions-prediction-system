from keras.models import load_model
import pandas as pd
from sklearn import preprocessing



class NNPredictor():
    def __init__(self):
        self.model = load_model('model_20190117.h5')


    def get_predict(self, param_list):
        # implement the model based predictor
        all_df = pd.read_csv("trainnosu.csv")
        cols=['suicide', 'sexuality', 'age', 'Education', 'rank', 'militaryservice', 'Time','testResult','hadSuicideMessage', 'confirmedDisease','emotionalProblems', 'mentalillness', 'familySuicideHistory','familyMembers', 'workplacePressure', 'EconomicIssues','personalPressure']
        all_df = all_df[cols]
        df = pd.DataFrame([list(param_list)],columns = [
        'suicide', 'sexuality', 'age', 'Education',
        'rank', 'militaryservice', 'Time','testResult',
        'suicideHistory', 'hadSuicideMessage', 'confirmedDisease',
        'emotionalProblems', 'mentalillness', 'familySuicideHistory',
        'familyMembers', 'workplacePressure', 'EconomicIssues','personalPressure'
        ])
        all_df = pd.concat([all_df,df])
        # scaledFeatures, _ = preprocess_data(df)
        scaledFeatures, _ = self.preprocess_data(all_df)
        result = self.model.predict(scaledFeatures[-1:])
        print(scaledFeatures)
        return result[0]


    def preprocess_data(self, data_df):
        pr_df = data_df.copy()
        pr_df['sexuality'] = data_df['sexuality'].astype('category').cat.codes
        pr_df['Education'] = data_df['Education'].astype('category').cat.codes
        pr_df['rank'] = data_df['rank'].astype('category').cat.codes
        pr_df['militaryservice'] = data_df['militaryservice'].astype('category').cat.codes
        pr_df['Time'] = data_df['Time'].astype('category').cat.codes
        pr_df['testResult'] = data_df['testResult'].astype('category').cat.codes

        ndarray = pr_df.values
        Features = ndarray[:,1:]
        Label = ndarray[:,0]
        minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaledFeatures = minmax_scale.fit_transform(Features)
        
        return scaledFeatures, Label
