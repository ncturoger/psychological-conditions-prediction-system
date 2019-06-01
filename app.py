from flask import Flask, request, jsonify, render_template, redirect, url_for
from predictors.NNPredictor import NNPredictor
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
        with app.config["graph"].as_default():
            result = app.config["predictor"].get_predict(parameter)
        
        if float(str(result[0])) > 0.5:
            return redirect(url_for('result_1'))
        
        else:
            return redirect(url_for('result_2'))
    else:
        return


if __name__ == "__main__":
    print("server start...")
    app.config["predictor"] = NNPredictor()
    app.config["graph"] = tf.get_default_graph()
    app.run(host="0.0.0.0", debug=False)

    