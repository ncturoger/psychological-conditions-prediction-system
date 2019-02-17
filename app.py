from flask import Flask, request, jsonify, render_template
import random
import keras
from keras.models import load_model
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf

app = Flask(__name__)
model = None
graph = tf.get_default_graph()

# jack = [0,'male',25,'u','officer','v','S','F',0,0,0,0,0,0,0,1,1,1]

# scaledFeatures, Label = preprocess_data(jack_df)
# a = model.predict(scaledFeatures)
# print(a)

# flask run --host=0.0.0.0
@app.route('/')
def hello_world():
    return render_template('home.html') 
    # return "12"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        sexuality = request.form.get('sexuality')
        age = request.form.get('age')
        education = request.form.get('education')
        rank = request.form.get('rank')
        militaryservice = request.form.get('militaryservice')
        time = request.form.get('time')
        testResult = request.form.get('testResult')
        suicideHistory = request.form.get('suicideHistory')
        hadSuicideMessage = request.form.get('hadSuicideMessage')
        confirmedDisease = request.form.get('confirmedDisease')
        emotionalProblems = request.form.get('emotionalProblems')
        mentalillness = request.form.get('mentalillness')
        familySuicideHistory = request.form.get('familySuicideHistory')
        familyMembers = request.form.get('familyMembers')
        workplacePressure = request.form.get('workplacePressure')
        EconomicIssues = request.form.get('EconomicIssues')
        personalPressure = request.form.get('personalPressure')
        parameter = [
            0,
            sexuality,
            age,
            education,
            rank,
            militaryservice,
            time,
            testResult,
            suicideHistory,
            hadSuicideMessage,
            confirmedDisease,
            emotionalProblems,
            mentalillness,
            familySuicideHistory,
            familyMembers,
            workplacePressure,
            EconomicIssues,
            personalPressure
        ]
        print(parameter)

        global graph
        with graph.as_default():
            result = get_predict(parameter)
        return jsonify({
            'name': "test",
            'result': str(result[0])
        })
        # return jsonify({
        #     "sss": str(name),
        #     'result': "ok"
        # })
    else:
        return


def get_predict(param_list):
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
    scaledFeatures, _ = preprocess_data(all_df)
    result = model.predict(scaledFeatures[-1:])
    print(scaledFeatures)
    return result[0]


def preprocess_data(data_df):
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

def server_load_model():
    global model
    model = load_model('model_20190117.h5')

if __name__ == "__main__":
    print("server start...")
    server_load_model()
    app.run(host="0.0.0.0", debug=False)

    