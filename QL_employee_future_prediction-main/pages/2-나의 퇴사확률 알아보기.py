import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from sys import path
path.append('module/')
import chu_funcs as chu
from yoon_funcs import new_e, new_t
import os

df = pd.read_csv("data/Employee.csv")

st.set_page_config(layout="centered")

st.title('이 회사에서 당신의 퇴사 확률은...?')

feat_list = df.columns.tolist()
feat_kor_list = ['학력', '입사 시기', '지점', '연봉 등급', '연령', '성별', '1개월 이상 업무 배제 경험', '업계 내 경력 기간']
feature_dict = {}
feature_kor_dict = {}
col1, col2 = st.columns(2)

for key, kor in zip(feat_list, feat_kor_list):
    feature_dict[key] = []
    feature_kor_dict[key] = kor
with col1:
    feature_dict["Education"].append(st.select_slider(feature_kor_dict["Education"], ['Bachelors', 'Masters', 'PHD']))
    feature_dict["JoiningYear"].append(st.slider(feature_kor_dict["JoiningYear"], 2010, 2022))
    feature_dict["City"].append(st.select_slider(feature_kor_dict["City"], ['Bangalore', 'Pune', 'New Delhi']))
    feature_dict["PaymentTier"].append(st.select_slider(feature_kor_dict["PaymentTier"], [1, 2, 3]))
with col2:
    feature_dict["Age"].append(st.slider(feature_kor_dict["Age"], 20, 50))
    feature_dict["Gender"].append(st.select_slider(feature_kor_dict["Gender"], ['Male', 'Female']))
    feature_dict["EverBenched"].append(st.select_slider(feature_kor_dict["EverBenched"], ['No', 'Yes']))
    feature_dict["ExperienceInCurrentDomain"].append(st.slider(feature_kor_dict["ExperienceInCurrentDomain"], 0, 10))

resp_data = pd.DataFrame(feature_dict)

st.write("퇴사 확률에 예측에 사용될 변수")
st.dataframe(resp_data.rename(columns = feature_kor_dict, index = {0: "현재 선택값"}))



if st.button("예측하기"):
    ## chu's model
    def Chu_predict():
        
        # X_train, X_test, y_train, y_test= chu.pre_df(df)

        # dtc = DecisionTreeClassifier(criterion= 'gini', max_depth= 8, min_samples_leaf= 4, min_samples_split= 2,random_state=42)

        # lrc = LogisticRegression(random_state=42)

        # rfc = RandomForestClassifier(criterion= 'entropy', max_depth= 8, n_estimators= 500, random_state=42)

        # adc = AdaBoostClassifier(learning_rate = 0.001, n_estimators= 180, random_state=42)

        # etc = ExtraTreesClassifier(criterion= 'gini', n_estimators= 70, random_state=42) 

        # gbdt = GradientBoostingClassifier(learning_rate= 0.1, loss= 'deviance', max_depth= 3, n_estimators= 150, random_state=42)

        # xgb = XGBClassifier(eval_metric= 'mlogloss', learning_rate= 1, n_estimators= 20, random_state=42) 

        # lgbm= LGBMClassifier(learning_rate= 0.1, max_bin= 100, n_estimators= 50, num_leaves= 10, random_state=42) 


        # models = {
        #     'chu_DT': dtc, 
        #     'chu_LR': lrc, 
        #     'chu_RF': rfc, 
        #     'chu_AdaBoost': adc, 
        #     'chu_ETC': etc,
        #     'chu_GBDT':gbdt,
        #     'chu_xgb':xgb,
        #     'chu_lgbm':lgbm,

        # }
        # chu_mod_accuracy = chu.result_table(models, X_train, y_train, X_test, y_test)[["Algorithm", "Accuracy"]]
    


        ### preprocessing new data
        new_df =chu.pre_new_data(df,resp_data)
        # new_df['LeaveOrNot'] = np.nan

               ## load fitted model
        file_path = "mod_obj/"
        files = []
        for i in os.listdir(file_path):
            if i.startswith("Chu_"):
                files.append(i)

        model_dict = {"Algorithm" : [],
        "Prediction" : [],
        "Accuracy" : [],
        "Weight" : []}  

        for file in files:
            model_dict["Algorithm"].append(file.split("_0.")[0])
            model_dict["Accuracy"].append(float(file.split("_")[-1].split(".joblib")[0]))            
            model = joblib.load(f'{file_path + file}')
            model_dict["Prediction"].append(int(model.predict(new_df)))
            model_dict["Weight"].append(0)

        return pd.DataFrame(model_dict)

        # ### prediction
        # chu_pred_results = pd.merge(chu.new_data_prediction(models, new_df, X_train, y_train, X_test, y_test), chu_mod_accuracy) 
        # return chu_pred_results

    ## Lee's model
    def Lee_predict():
        file_path = "mod_obj/"
        files = []
        for i in os.listdir(file_path):
            if i.startswith("Lee_"):
                files.append(i)
        

        ## preprocessing new data
        ohe_name = "ohe.joblib.gz"        
        fitted_ohe = joblib.load(f'{file_path + ohe_name}')
        cat_col = ["Education", "City", "PaymentTier", "Gender", "EverBenched"]
        num_col = resp_data.columns[~resp_data.columns.isin(cat_col)].tolist()
        X_test_ohe = pd.DataFrame(fitted_ohe.transform(resp_data[cat_col]).toarray(), columns = fitted_ohe.get_feature_names_out())
        X_test_ohe[num_col] = resp_data[num_col].copy()

        ## load fitted model
        file_path = "mod_obj/"
        files = []
        for i in os.listdir(file_path):
            if i.startswith("Lee_"):
                files.append(i)

        model_dict = {"Algorithm" : [],
        "Prediction" : [],
        "Accuracy" : [],
        "Weight" : []}  

        for file in files:
            model_dict["Algorithm"].append(file.split("_0.")[0])
            model_dict["Accuracy"].append(float(file.split("_")[-1].split(".joblib")[0]))            
            model = joblib.load(f'{file_path + file}')
            model_dict["Prediction"].append(int(model.predict(X_test_ohe)))
            model_dict["Weight"].append(0)

        return pd.DataFrame(model_dict)

    ## Yoon's models
    def Yoon_predict():
        ## preprocessing new data
        test_data = new_t(resp_data)
        
        ## load fitted model
        file_path = "mod_obj/"
        files = []
        for i in os.listdir(file_path):
            if i.startswith("Yoon_"):
                files.append(i)

        model_dict = {"Algorithm" : [],
        "Prediction" : [],
        "Accuracy" : [],
        "Weight" : []}  

        for file in files:
            model_dict["Algorithm"].append(file.split("_0.")[0])
            model_dict["Accuracy"].append(float(file.split("_")[-1].split(".joblib")[0]))            
            model = joblib.load(f'{file_path + file}')
            model_dict["Prediction"].append(int(model.predict(test_data)))
            model_dict["Weight"].append(0)
        

        return pd.DataFrame(model_dict)


    ## showing results
    
    model_results = pd.concat([Chu_predict(), Lee_predict(), Yoon_predict()]).reset_index(drop = True)
    model_results['Weight']=model_results['Accuracy']/model_results['Accuracy'].sum()
    
    n_models = len(model_results["Algorithm"])
    leave_prob = np.sum(model_results["Prediction"] * model_results["Weight"])

    st.subheader(f'{n_models}개의 모델로 예측한 결과,')
    st.metric(label="퇴사 확률은", value= f"{leave_prob * 100:.3f}%")

    col1, col2, col3 = st.columns([1, 10, 1])
    with col1:
        st.write(' ')

    with col2:
        if leave_prob >= 0.5:
            st.image('img/out.jpeg')
        else:
            st.image('img/in.png')

    with col3:
        st.write(' ')

    with st.expander("각 모델 예측 결과 확인"):
        st.write(model_results)
        st.write("Weighted Sum of Prediction = " +
        ' + '.join([f'({p} * {w})' for p, w in zip(model_results["Prediction"], model_results["Weight"])]) +
        f' = {leave_prob:.5f}')

    


