from flask import Flask, request, jsonify, render_template, redirect, url_for
from predictors.NNPredictor import NNPredictor
from predictors.XGBoostPredictor import XGBoostPredictor
import tensorflow as tf

app = Flask(__name__)

# jack = [0,'male',25,'u','officer','v','S','F',0,0,0,0,0,0,0,1,1,1]

# scaledFeatures, Label = preprocess_data(jack_df)
# a = model.predict(scaledFeatures)
# print(a)

# flask run --host=0.0.0.0
@app.route('/')
def home_page():
    return render_template('home.html') 

@app.route('/form')
def fillform():
    if app.config["PREDICTOR_METHOD"] == "XGBoostPredictor":
        return render_template('form_v3.html')
    
    elif app.config["PREDICTOR_METHOD"] == "NNPredictor":
        return render_template('form_v2.html')

@app.route('/result_positive')
def result_1():
    return render_template('result1.html') 

@app.route('/result_negative')
def result_2():
    return render_template('result2.html') 

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if app.config["PREDICTOR_METHOD"] == "XGBoostPredictor":
            cols = [
                    '01.sui_history', '02.had_sui_message',
                    '03.disease', '04.sub_abuse', '05.Gender_disorder',
                    '06.mtl_illness', '07.fam_sui_history', '08.emo_imbalance',
                    '09.rej_service', '10.relationship issue', '11.fam_problem',
                    '12.inter_problem', '13.social_support', '14.maladaptive',
                    '15.wk_pressure', '16.mng_problem', '17.setbacks',
                    '18.punish&jud_case', '19.eco_burden', '20.debt'
                    ]

            parameter = [int(request.form.get(col)) for col in cols]
            result = app.config["predictor"].get_predict(parameter)
            
            if result == 1:
                return redirect(url_for('result_1'))
            
            else:
                return redirect(url_for('result_2'))

        elif app.config["PREDICTOR_METHOD"] == "NNPredictor":
            cols = [
                    'sexuality', 'age', 'Education', 'rank',
                    'militaryservice', 'Time','testResult','suicideHistory',
                    'hadSuicideMessage','confirmedDisease','emotionalProblems',
                    'mentalillness','familySuicideHistory','familyMembers',
                    'workplacePressure','EconomicIssues','personalPressure'
                   ]
            
            parameter = [0]
            for col in cols:
                parameter.append(request.form.get(col))

            with app.config["graph"].as_default():
                result = app.config["predictor"].get_predict(parameter)

            if float(result) >= 0.5:
                return redirect(url_for('result_1'))
            
            else:
                return redirect(url_for('result_2'))

    else:
        return


if __name__ == "__main__":
    print("server start...")
    app.config.from_pyfile('config.py')
    if app.config["PREDICTOR_METHOD"] == "XGBoostPredictor":
        app.config["predictor"] = XGBoostPredictor()
        print("Use XGBoostPredictor")
    
    elif app.config["PREDICTOR_METHOD"] == "NNPredictor":
        app.config["graph"] = tf.get_default_graph()
        app.config["predictor"] = NNPredictor()
        print("Use NNPredictor")
    
    app.run(host="0.0.0.0", debug=False)

    