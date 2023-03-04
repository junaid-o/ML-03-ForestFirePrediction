import pickle
import pandas as pd
import numpy as np
import matplotlib
from batch_prediction import batch_prediction
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template, flash, redirect
#from flask_paginate import Pagination, get_page_parameter

#from flask_cors import cross_origin
#######################################################
def styler(df):
    df_styler = df.style.set_properties(**{'font-size': '20px', 'font-weight': 'bold'}) \
        .background_gradient(cmap='Blues', subset=['score']) \
        .format({'score': '{:.2f}'}) \
        .hide_index()
    return df_styler
############### MODELS IMPORT ####################
#class_mdoel_1 = pickle.load(open('Models/Classification_models/tuned_models_raw4_3/tuned_RandomForestClassifier_model.pkl','rb'))
#class_mdoel_2 = pickle.load(open('Models/Classification_models/tuned_models_raw4_3_without_month/tuned_RandomForestClassifier_model.pkl','rb'))
class_mdoel_voting_1 = pickle.load(open('Models/Classification_models/VotingClassifier_raw4_3/VotingClassifier_raw4_3_model.pkl','rb'))
class_mdoel_voting_2 = pickle.load(open('Models/Classification_models/VotingClassifier_raw4_3_without_month/VotingClassifier_raw4_3_without_month_model.pkl','rb'))


####    Regression Models Import  #####
#reg_mdoel_1 = pickle.load(open('Models/Regression_models/tuned_models_raw4_4/tuned_RandomForestRegressor_model.pkl','rb'))
#reg_mdoel_2 = pickle.load(open('Models/Regression_models/tuned_models_raw4_6/tuned_RandomForestRegressor_model.pkl','rb'))
#reg_mdoel_3 = pickle.load(open('Models/Regression_models/tuned_models_raw4_5/tuned_RandomForestRegressor_model.pkl','rb'))

reg_mdoel_1 = pickle.load(open('Models/Regression_models/VotingClassifier_raw4_4/VotingClassifier_raw4_4_model.pkl','rb'))
reg_mdoel_2 = pickle.load(open('Models/Regression_models/VotingClassifier_raw4_6/VotingClassifier_raw4_6_model.pkl','rb'))
reg_mdoel_3 = pickle.load(open('Models/Regression_models/VotingClassifier_raw4_5/VotingClassifier_raw4_5_model.pkl','rb'))
############### SCALAR IMPORT  ##########################

scalar_raw4_3 = pickle.load(open('./StandardScalar/classification_scaler_raw4_3.pkl','rb'))
scalar_reg_raw4_6 = pickle.load(open('./StandardScalar/regression_scaler_raw4_6.pkl','rb'))
scalar_reg_raw4_4 = pickle.load(open('./StandardScalar/regression_scaler_raw4_4.pkl','rb'))
scalar_reg_raw4_5 = pickle.load(open('./StandardScalar/regression_scaler_raw4_5.pkl', 'rb'))
#########################################
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':

        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':

                df = pd.read_csv(file)
                #print('File Uploaded. It is there')
                if 'FWI_prediction' in request.form:
                    print('EXecuting FWI prediction')
                    df = batch_prediction.batch_prediction.FWI_prediction(df=df, reg_model_1=reg_mdoel_1,
                                                           reg_model_2=reg_mdoel_2,
                                                           reg_model_3 = reg_mdoel_3)
                    message = '<h2>Prediction By EnsemblTechnique Using RandomForest, SVR</h2>'
                else:
                    print('executing FIRE prediciton')
                    df = batch_prediction.batch_prediction.FIRE_prediction(df=df,
                                                                           class_model_1=class_mdoel_voting_1,
                                                                           class_model_2 = class_mdoel_voting_2)
                    message = '<h2>Prediction By EnsemblTechnique Using RandomForest, LightGBM, AdaBoost and Gradient Boosting Trees</h2>'

                #print(df)
                df = df.style.set_properties(**{'font-size': '15px', 'font-weight': 'bold', 'border-color':'black', 'background-color': 'rgba(0, 0, 0, 0.1)'}) \
                    #.background_gradient(cmap='Greys') \
                    #subset=['FWI Prediction']
                df.set_table_styles(
                    [{'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '1px solid grey')]},
                     {'selector': 'th', 'props': [('border', '1px solid grey')]},
                     #{'selector': 'td',
                     # 'props': [('border', '0.5px solid grey'), ('text-align', 'center')]},
                     ])

                final_result = df.to_html(classes=['table', 'table-striped'],bold_rows=True, border=0.5)
                prediction_heading = '<h1>File Uploaded!<h1>'
                #message = ''
                # process data from file

        else:

            Temperature = float(request.form['Temperature'])
            RH = float(request.form['RH'])
            try:
                Rain = float(request.form['Rain'])
            except:
                Rain = request.form['Rain']
            try:
                Ws = float(request.form['WindSpeed'])
            except:
                Ws = request.form['WindSpeed']
            DC = float(request.form['DC'])
            FWI = request.form['FWI']
            FFMC = request.form['FFMC']
            DMC = request.form['DMC']
            ISI = request.form['ISI']
            BUI = request.form['BUI']
            month = request.form['month']
            # Do something with the input data

            ########## Feature Engineering: CLASSIFICATION & REGRESSION BOTH ########


            if 'FWI_prediction' in request.form:
                print('FWI_prediction' in request.form)
                if FFMC and BUI and ISI and DMC:
                    try:
                        BUI = float(BUI)
                        FFMC = float(FFMC)
                        ISI = float(ISI)
                        DMC = float(DMC)

                        if FFMC == 0 or BUI == 0 or ISI == 0 or DMC == 0:
                            message = 'ZeroDivisionError'
                            final_result = ''
                            prediction_heading = ''
                        else:
                            ISI_FFMC = ISI / FFMC
                            ISI_DMC_BUI = float(ISI) / float(DMC) * float(BUI)


                            if Ws and Rain:

                                ########### MAKING PREDICTION: Regression using Model1 #############
                                print('Executing Regressor 1')
                                df_reg_result1 = pd.DataFrame([Temperature, RH, Ws, Rain, ISI_FFMC, ISI_DMC_BUI]).T
                                df_reg_result1.columns = ['Temperature', 'RH', 'Ws', 'Rain', 'ISI/FFMC', 'ISI/DMC*BUI']

                                df_reg1 = pd.DataFrame([Temperature, RH, Ws, Rain, ISI_FFMC, ISI_DMC_BUI]).T
                                df_reg1.columns = ['Temperature', 'RH', 'Ws', 'Rain', 'ISI/FFMC', 'ISI/DMC*BUI']

                                df_reg_scaled1 = scalar_reg_raw4_4.transform(df_reg1)

                                df_reg_scaled1 = pd.DataFrame(df_reg_scaled1, columns=df_reg1.columns)
                                prediction_reg1 = reg_mdoel_1.predict(df_reg_scaled1)
                                # df_reg_result.insert(4, 'Predicted FWI', prediction_reg1)
                                # print(prediction_reg1)

                            ############## MAKING PRDICTIONS: Regression Using Model 2 ###############

                            print('Starting Regressor 2')
                            df_reg_result2 = pd.DataFrame([Temperature, RH, ISI_FFMC, ISI_DMC_BUI]).T
                            df_reg_result2.columns = ['Temperature', 'RH', 'ISI/FFMC', 'ISI/DMC*BUI']

                            df_reg2 = pd.DataFrame([Temperature, RH, ISI_FFMC, ISI_DMC_BUI]).T
                            df_reg2.columns = ['Temperature', 'RH', 'ISI/FFMC', 'ISI/DMC*BUI']

                            df_reg_scaled2 = scalar_reg_raw4_6.transform(df_reg2)

                            df_reg_scaled2 = pd.DataFrame(df_reg_scaled2, columns=df_reg2.columns)

                            prediction_reg2 = reg_mdoel_2.predict(df_reg_scaled2)
                            #print('Prediction from second Regressor ',prediction_reg2)

                            ########## MAKING PREDICTION: Regression using MODEL3 ###########
                            print('Starting Regressor 3')
                            df_reg_result3 = pd.DataFrame([Temperature, RH, Ws, Rain, DC, ISI_FFMC, ISI_DMC_BUI]).T
                            df_reg_result3.columns = ['Temperature', 'RH', 'Ws', 'Rain', 'DC', 'ISI/FFMC',
                                                      'ISI/DMC*BUI']
                            df_reg3 = pd.DataFrame([Temperature, RH, Ws, Rain, DC, ISI_FFMC, ISI_DMC_BUI]).T
                            df_reg3.columns = ['Temperature', 'RH', 'Ws', 'Rain', 'DC', 'ISI/FFMC', 'ISI/DMC*BUI']

                            df_reg_scaled3 = scalar_reg_raw4_5.transform(df_reg3)
                            df_reg_scaled3 = pd.DataFrame(df_reg_scaled3, columns=df_reg3.columns)
                            prediction_reg3 = reg_mdoel_3.predict(df_reg_scaled3)

                            try:
                                prediction_reg = (prediction_reg1 + prediction_reg2 + prediction_reg3) / 3
                                df_reg_result1.insert(len(df_reg_result1.columns), 'Predicted FWI', prediction_reg)

                                final_result = df_reg_result1.to_html(max_cols=9)
                                prediction_heading = '<h1>Prediction</h1>'
                                message =   f"""
                                             <br>Prdiction Based on Multiple Regressor<br><br>
                                            <h1 style='color:yellow; text-align:center'><span style='background-color:rgb(0,0,0,0.8)'>FWI = {prediction_reg[0]}</h1>
                                            """
                                #print('Prdiction Based on Multiple Regressor')

                            except Exception as e:
                                prediction_reg = prediction_reg2
                                df_reg_result2.insert(len(df_reg_result2.columns), 'Predicted FWI', prediction_reg)
                                final_result = df_reg_result2.to_html(max_cols=9)
                                prediction_heading = '<h1>Prediction</h1>'
                                message =   f"""
                                             <br>Prdiction Based on Single Regressor<br><br>
                                            <h1 style='color:yellow; text-align:center'><span style='background-color:rgb(0,0,0,0.8)'>FWI = {prediction_reg[0]}</h1>
                                            """
                                #print(e)
                                #print('Prediction from One Regressor only')
                    except:
                        message = """<h3 style='text-align:center'>Values Can't Be Empty</h3>"""
                        final_result = ''
                        prediction_heading = """
                                            <p><H1 style="text-align: center;font-size:3em; color:red">
                                            Warning!<H1>
                                            <H1>"""

            elif FWI and FFMC and BUI and ISI:
                try:
                    FWI = float(FWI)
                    BUI = float(BUI)
                    FFMC = float(FFMC)
                    ISI = float(ISI)
                    DMC = float(DMC)
                    if FWI == 0 or FFMC == 0 or BUI == 0 or ISI == 0:
                        message = """<h3 style='text-align:center'>Zero Is Not Allowed</h3>"""
                        final_result = ''
                        prediction_heading = """
                                            <p><H1 style="text-align: center;font-size:3em; color:red">
                                            Warning!<H1>
                                            <H1>"""
                    elif not (Rain and Ws):
                        message = """<h3 style= 'text-align:center'>FWI, Wind Speed and Rain are mandatory for Fire prediction</h3>"""
                        final_result = ''
                        prediction_heading = """
                                            <p><H1 style="text-align: center;font-size:3em; color:red">
                                            Warning!<H1>
                                            <H1>"""
                    else:
                        try:

                            DMC_FWI_ISI = (DMC / FWI) / ISI
                            FWI_FFMC = FWI / FFMC
                            FWI_BUI = FWI / BUI
                            print('executed', FWI)

                            ### Preparing Input Data: CLASSIFICATION  #######
                            try:
                                if month == '7':
                                    month_07 = 1
                                    month_08 = 0
                                    month_09 = 0
                                if month == '8':
                                    month_07 = 0
                                    month_08 = 1
                                    month_09 = 0
                                if month == '9':
                                    month_07 = 0
                                    month_08 = 0
                                    month_09 = 1

                                #print([month_07,month_08,month_09])
                                print('Prdicting All the featurs Including Month')

                                df_class_result = pd.DataFrame([month_07, month_08, month_09, Temperature, RH, Rain, DC, FWI]).T

                                df_class_result.columns = ['month_07', 'month_08', 'month_09', 'Temperature', 'RH', 'Rain',
                                                           'DC', 'FWI']

                                df_class = pd.DataFrame(
                                    [month_07, month_08, month_09, Temperature, RH, Rain, DC, FWI, FWI_FFMC, DMC_FWI_ISI,
                                     FWI_BUI]).T
                                df_class.columns = ['month_07', 'month_08', 'month_09', 'Temperature', 'RH', 'Rain', 'DC',
                                                    'FWI',
                                                    'FWI/FFMC', '(DMC/FWI)/ISI', 'FWI/BUI']


                                df_class_scaled = scalar_raw4_3.transform(
                                    df_class.drop(['month_07', 'month_08', 'month_09'], axis=1))
                                print('Transformation DOne')
                                df_class_scaled = pd.DataFrame(df_class_scaled,
                                                               columns=df_class.drop(['month_07', 'month_08', 'month_09'],
                                                                                     axis=1, ).columns)
                                df_class_scaled = pd.concat([df_class[['month_07', 'month_08', 'month_09']], df_class_scaled],
                                                            axis=1)

                                ############## MAKING PRDICTIONS: CLASSIFICATION ###############
                                prediction_class = class_mdoel_voting_1.predict(df_class_scaled)
                                class_proba = class_mdoel_voting_1.predict_proba(df_class_scaled)
                                # pd.DataFrame(prediction, columns=['Fire Proba', 'Not Fire Proba']).to_html()
                                df_class_result.insert(len(df_class_result.columns), 'NotFire_Prediction', prediction_class)
                                #df_class_result.insert(len(df_class_result.columns), 'NotFire_Prediction', prediction_class)
                                print('PREDICTION Done')
                                if prediction_class[0] == 0:
                                    pred_class = 'Fire'
                                else:
                                    pred_class = 'Not Fire'
                            except:
                                print('Predicting Fire Class Without Month')
                                df_class_result = pd.DataFrame([Temperature, RH, Rain, DC, FWI]).T

                                df_class_result.columns = ['Temperature', 'RH', 'Rain','DC', 'FWI']

                                df_class = pd.DataFrame(
                                    [Temperature, RH, Rain, DC, FWI, FWI_FFMC, DMC_FWI_ISI, FWI_BUI]).T
                                df_class.columns = ['Temperature', 'RH', 'Rain', 'DC', 'FWI','FWI/FFMC', '(DMC/FWI)/ISI', 'FWI/BUI']

                                df_class_scaled = scalar_raw4_3.transform(df_class)
                                #print('Transformation Done')
                                df_class_scaled = pd.DataFrame(df_class_scaled,
                                                               columns=df_class.columns)

                                ############## MAKING PRDICTIONS: CLASSIFICATION ###############
                                prediction_class = class_mdoel_voting_2.predict(df_class_scaled)
                                class_proba = class_mdoel_voting_2.predict_proba(df_class_scaled)
                                # pd.DataFrame(prediction, columns=['Fire Proba', 'Not Fire Proba']).to_html()
                                df_class_result.insert(len(df_class_result.columns), 'NotFire_Prediction', prediction_class)

                                print('PREDICTION Done')
                                if prediction_class[0] == 0:
                                    pred_class = 'Fire'
                                else:
                                    pred_class = 'Not Fire'


                            final_result = df_class_result.to_html(escape=False, index=True, header=True, max_cols=9)
                            prediction_heading = """<h1 style='text-align: left'>Prediction</h1>"""
                            message = f"""<br><p style='color:yellow;'><span style= 'background-color: rgb(0,0,0,0.8)'>Successful!</p>
                                        <br> Predicted By Ensemble Technique Using RandomForest,AdaBoost, Gradient Boosting Trees and LightGBM
                                        <br><h1 style='text-align:center;color: yellow;'><span style='background-color: rgb(0,0,0,0.7)'>{pred_class}
                                        <br>Probability = {np.max(class_proba, axis=1)[0] * 100}%</h1>
                                        """
                        except ZeroDivisionError:
                            message = """<h3 style= 'text-align:center'>Zero Is Not Allowed</h3>"""
                            final_result = ''
                            prediction_heading = """
                                                <p><H1 style="text-align: center;font-size:3em; color:red">
                                                Warning!<H1>
                                                <H1>"""
                        except:
                            message = """<h3 style= 'text-align:center'>An Error Occured</h3>"""
                            final_result = ''
                            prediction_heading = """
                                                <p><H1 style="text-align: center;font-size:3em; color:red">
                                                Oops!<H1>
                                                <H1>"""
                except ValueError:
                    message = """<h3 style= 'text-align:center'>Mandatory Fields Can't Be Empty.<br>
                                Rain and Wind speed are optional for FWI prediciton but is needed for Fire prediction</h3>"""
                    final_result = ''
                    prediction_heading = """
                                        <p><H1 style="text-align: center;font-size:3em; color:red">
                                        Warning!<H1>
                                        <H1>"""
            else:
                message = 'FWI, FFMC, BUI and ISI are mandatory for Fire prediction'
                final_result = ''
                prediction_heading = """
                                    <p><H1 style="text-align: center;font-size:3em; color:red">
                                    Warning!<H1>
                                    <H1>"""

    #return render_template('home.html', prediction=final_result, PredictionHeading=prediction_heading, message=message)
    return render_template('result.html', prediction=final_result, PredictionHeading=prediction_heading, message=message)





if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8001, debug=True)
    #app.run(debug=True)
    #app.run(host='0.0.0.0/0', debug=True)  # for hosting


