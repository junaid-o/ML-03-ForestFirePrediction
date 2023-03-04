import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
class batch_prediction:
    def FWI_prediction(df, reg_model_1, reg_model_2, reg_model_3):
        """
        Batch_Prediction Can be used for making FWI predition on
        Entire DataSet But data must be clean and in proper format.
        A dataframe o be used for FWI prediction must have only columns.
        Each column should have header name Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI
        """

        scalar_reg_raw4_4 = pickle.load(open('./StandardScalar/regression_scaler_raw4_4.pkl', 'rb'))
        scalar_reg_raw4_6 = pickle.load(open('./StandardScalar/regression_scaler_raw4_6.pkl', 'rb'))
        scalar_reg_raw4_5 = pickle.load(open('./StandardScalar/regression_scaler_raw4_5.pkl', 'rb'))

        #df = df.drop(['Date', 'day', 'year', 'month', 'Classes'], axis=1,inplace=False)

        #df = df[['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI']]
        try:
            df = df[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']]

            ##### PREDICTION BY MODEL3 #####
            df3 = df.copy()
            df3['ISI/FFMC'] = df3['ISI'] / df3['FFMC']
            df3['ISI/DMC*BUI'] = df3['ISI'] / df3['DMC'] / df3['BUI']
            df3.drop(['FFMC', 'DMC', 'ISI', 'BUI'], axis=1, inplace=True)

            df_scaled_3 = scalar_reg_raw4_5.transform(df3)
            df_reg_result3 = pd.DataFrame(df_scaled_3, columns= df3.columns)
            prediction_reg_3 = reg_model_3.predict(df_reg_result3)

            if 'Rain' and 'Ws' in list(df.columns):
                df1 = df
                df1['ISI/FFMC'] = df1['ISI'] / df1['FFMC']
                df1['ISI/DMC*BUI'] = df1['ISI'] / df1['DMC'] / df1['BUI']
                df1.drop(['FFMC', 'DMC', 'DC', 'ISI', 'BUI'], axis=1, inplace=True)
                try:
                    df.drop(['FWI'], axis=1, inplace=True)
                except:
                    pass
                #### MODEL PREDICTION ##########
                df_scaled_1 = scalar_reg_raw4_4.transform(df1)
                df_reg_result1 = pd.DataFrame(df_scaled_1, columns=df1.columns)
                prediction_reg_1 = reg_model_1.predict(df_reg_result1)
                #df_orig.insert(4, 'FWI Prediction', prediction_1)
                #print(df_reg_result1)
                #### MODEL PREDICTION ##########
                df2 = df.drop(['Rain','Ws'], axis=1)
                df_scaled_2 = scalar_reg_raw4_6.transform(df2)
                df_reg_result2 = pd.DataFrame(df_scaled_2, columns=df2.columns)
                prediction_reg_2 = reg_model_2.predict(df_reg_result2)

                ############ COMBINING PREDICTIONS ###############

                prediction = (prediction_reg_1 + prediction_reg_2 +prediction_reg_3)/3
                df2.insert(len(df2.columns), 'FWI Prediction', prediction)
                #print(df_reg_result2)
        except:

            ########## MAIKING PREDICTION IF RAIN AND WIND SPPEED IS NOT GIVEN ##############

            df2 = df[['Temperature', 'RH','FFMC', 'DMC', 'DC', 'ISI', 'BUI']]
            df2['ISI/FFMC'] = df2['ISI'] / df2['FFMC']
            df2['ISI/DMC*BUI'] = df2['ISI'] / df2['DMC'] / df2['BUI']
            df2.drop(['FFMC', 'DMC', 'DC', 'ISI', 'BUI'], axis=1, inplace=True)
            df_scaled_2 = scalar_reg_raw4_6.transform(df2)
            df_reg_result2 = pd.DataFrame(df_scaled_2, columns=df2.columns)
            prediction_reg2 = reg_model_2.predict(df_reg_result2)
            df2.insert(len(df2.columns),'FWI Prediction',prediction_reg2)
            #pass
        return df2

    def FIRE_prediction(df, class_model_1, class_model_2):

        scalar_raw4_3 = pickle.load(open('./StandardScalar/classification_scaler_raw4_3.pkl', 'rb'))
        if 'month' in list(df.columns):
            df['month'] = df['month'].astype('object')
            df1 = df[['month', 'Temperature', 'RH', 'Rain','DC','DMC', 'FWI', 'FFMC','BUI','ISI']]
            df1 = pd.get_dummies(df1)
            print('get Dummies',df1)
            df1 = df1[['month_7', 'month_8', 'month_9', 'Temperature', 'RH', 'Rain','DC', 'FWI', 'FFMC','DMC','BUI','ISI']]
            print(df1.columns)

            df1['FWI_FFMC'] = df1['FWI'] / df1['FFMC']
            df1['DMC_FWI_ISI'] = df1['DMC'] /df1['FWI'] / df1['ISI']
            df1['FWI_BUI'] = df1['FWI'] / df1['BUI']
            df1.rename(columns={'FWI_FFMC':'FWI/FFMC',
                                'DMC_FWI_ISI':'(DMC/FWI)/ISI',
                                'FWI_BUI':'FWI/BUI'}, inplace=True)

            ############ Handling NaN and Inf Values ##########
            df1['(DMC/FWI)/ISI'].replace(to_replace=np.inf, value=np.nan, inplace=True)
            df1['(DMC/FWI)/ISI'].replace(to_replace=np.nan, value=df1['(DMC/FWI)/ISI'].mean(), inplace=True)
            df1['FWI/BUI'].replace(to_replace='inf', value=np.nan, inplace=True)
            df1['FWI/BUI'].replace(to_replace=np.nan, value=df1['FWI/BUI'].mean(), inplace=True)
            df1['FWI/FFMC'].replace(to_replace='inf', value=np.nan, inplace=True)
            df1['FWI/FFMC'].replace(to_replace=np.nan, value=df1['FWI/FFMC'].mean(), inplace=True)
            ##########################################################

            df1.drop(['FFMC', 'DMC', 'ISI', 'BUI'], axis=1, inplace=True)
            df_scaled_1 = scalar_raw4_3.transform(df1.drop(['month_7', 'month_8', 'month_9'],axis=1))
            #print(df1.columns)
            df_class_result1 = pd.DataFrame(df_scaled_1, columns=df1.drop(['month_7','month_8','month_9'],axis=1,).columns)
            df_class_result1 = pd.concat([df1[['month_7', 'month_8', 'month_9']].reset_index(drop=True), df_class_result1], axis=1)
            #print(df_class_result1)
            prediction_class_1 = class_model_1.predict(df_class_result1)
            #class_proba1 = class_model_2.best_estimator_.predict_proba(df_class_result1)

            df.insert(len(df.columns), 'Not_FIRE_Pred', prediction_class_1)
        else:
            ############## MAKING PREDICTION BY MODEL 2WITHOUT MONTH ###########
            df1 = df[['Temperature', 'RH', 'Rain','DC','DMC', 'FWI', 'FFMC','BUI','ISI']]
            df1 = pd.get_dummies(df1)
            print('get Dummies',df1)
            df1 = df1[['Temperature', 'RH', 'Rain','DC', 'FWI', 'FFMC','DMC','BUI','ISI']]
            print(df1.columns)

            df1['FWI_FFMC'] = df1['FWI'] / df1['FFMC']
            df1['DMC_FWI_ISI'] = df1['DMC'] /df1['FWI'] / df1['ISI']
            df1['FWI_BUI'] = df1['FWI'] / df1['BUI']
            df1.rename(columns={'FWI_FFMC':'FWI/FFMC',
                                'DMC_FWI_ISI':'(DMC/FWI)/ISI',
                                'FWI_BUI':'FWI/BUI'}, inplace=True)

            ############ Handling NaN and Inf Values ##########
            df1['(DMC/FWI)/ISI'].replace(to_replace=np.inf, value=np.nan, inplace=True)
            df1['(DMC/FWI)/ISI'].replace(to_replace=np.nan, value=df1['(DMC/FWI)/ISI'].mean(), inplace=True)
            df1['FWI/BUI'].replace(to_replace='inf', value=np.nan, inplace=True)
            df1['FWI/BUI'].replace(to_replace=np.nan, value=df1['FWI/BUI'].mean(), inplace=True)
            df1['FWI/FFMC'].replace(to_replace='inf', value=np.nan, inplace=True)
            df1['FWI/FFMC'].replace(to_replace=np.nan, value=df1['FWI/FFMC'].mean(), inplace=True)

            #df1.drop(['month_7', 'month_8', 'month_9'], axis=1, inplace=True)
            df_scaled_1 = scalar_raw4_3.transform(df1)
            #print(df1.columns)
            df_class_result1 = pd.DataFrame(df_scaled_1, columns=df1.columns)
            #print(df_class_result1)
            prediction_class_1 = class_model_2.predict(df_class_result1)
            #class_proba2 = class_model_2.best_estimator_.predict_proba(df_class_result2)
            df.insert(len(df.columns), 'Not_FIRE_Pred', prediction_class_1)

        return df
