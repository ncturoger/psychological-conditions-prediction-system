
import joblib
from xgboost import XGBClassifier
import pandas as pd

class XGBoostPredictor():
    def __init__(self):
        self.model = joblib.load("XGBoost_model_20190601.m")
    
    def get_predict(self, param_list):
        cols = [
        '01.sui_history', '02.had_sui_message',
        '03.disease', '04.sub_abuse', '05.Gender_disorder',
        '06.mtl_illness', '07.fam_sui_history', '08.emo_imbalance',
        '09.rej_service', '10.relationship issue', '11.fam_problem',
        '12.inter_problem', '13.social_support', '14.maladaptive',
        '15.wk_pressure', '16.mng_problem', '17.setbacks',
        '18.punish&jud_case', '19.eco_burden', '20.debt'
        ]

        df = pd.DataFrame([list(param_list)],columns = cols)
        df.astype(int)
        print(df.dtypes)
        print(df)
        result = self.model.predict(df)
        return result[0]