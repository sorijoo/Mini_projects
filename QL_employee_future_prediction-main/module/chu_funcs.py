import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def pre_df(df):
    #중복값 없애기
    df = df.drop_duplicates().copy()
    
    #원핫 인코딩하기 - ['Education','City','Gender','EverBenched']
    ohe=pd.get_dummies(df[['Education','City','Gender','EverBenched']])
    ohe_list=ohe.columns.tolist()
    df[ohe_list]=pd.get_dummies(df[['Education','City','Gender','EverBenched']])
    #필요없는 컬럼 삭제하기
    df=df.drop(['Education','City','Gender','EverBenched', 'EverBenched_No','Gender_Female'], axis=1)
    
    #나이 qcut 하고 원핫 인코딩하기
    df['AgeGroup'] = pd.qcut(df['Age'], q=3, labels=['Young', 'Middle', 'Old'])
    age_ohe=pd.get_dummies(df.AgeGroup)
    age_list=age_ohe.columns.tolist()
    df[age_list]=age_ohe
    #Age 드랍하기
    df=df.drop(['Age','AgeGroup'],axis=1)
    
    label_name = "LeaveOrNot"
    
    X = df[df.columns[df.columns!=label_name]]
    y = df[label_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)
    
    return X_train, X_test, y_train, y_test




def train_classifier(models,X_train,y_train,X_test,y_test):
    models.fit(X_train,y_train)
    y_pred = models.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,zero_division=0)
    recall = recall_score(y_test,y_pred)
    
    return accuracy,precision,recall

def result_table(models, X_train, y_train, X_test, y_test):
    accuracy_scores = []
    precision_scores = []
    recall_scores=[]

    for name,model in models.items():
        accuracy,precision, recall= train_classifier(model, X_train,y_train,X_test,y_test)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)

    performance_df = pd.DataFrame({'Algorithm':models.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores, 'Recall': recall_scores})
    performance_df = performance_df.sort_values('Accuracy',ascending=False)
    
    return performance_df

def pre_new_data(original_df,new_df):
    #중복값 없애기
    df = original_df.drop_duplicates()
    
    # concat으로 받기
    df=pd.concat([df,new_df])
    
    #원핫 인코딩하기 - ['Education','City','Gender','EverBenched']
    ohe=pd.get_dummies(df[['Education','City','Gender','EverBenched']])
    ohe_list=ohe.columns.tolist()
    df[ohe_list]=pd.get_dummies(df[['Education','City','Gender','EverBenched']])
    #필요없는 컬럼 삭제하기
    df=df.drop(['Education','City','Gender','EverBenched', 'EverBenched_No','Gender_Female'], axis=1)
    
    #나이 qcut 하고 원핫 인코딩하기
    df['AgeGroup'] = pd.qcut(df['Age'], q=3, labels=['Young', 'Middle', 'Old'])
    age_ohe=pd.get_dummies(df.AgeGroup)
    age_list=age_ohe.columns.tolist()
    df[age_list]=age_ohe
    #Age 드랍하기
    df=df.drop(['Age','AgeGroup'],axis=1)
    
    #transpose 시키기
    new=pd.DataFrame(df.iloc[-1])
    new=new.transpose()
    
    new=new.dropna(axis=1)
    
    return new

def y_predicting(models,X_train,y_train,X_test,y_test):
    models.fit(X_train,y_train)
    y_pred = models.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)

    return y_pred[0], accuracy


def new_data_prediction(models, new, X_train, y_train, X_test, y_test):
    y_pred_results = []
    accuracy_scores = []

    for name,model in models.items():
        y_prediction, accuracy = y_predicting(model, X_train,y_train,X_test,y_test)
        y_pred_results.append(y_prediction)
        accuracy_scores.append(accuracy)

    predict_table=pd.DataFrame({'Algorithm': models.keys(), 'Prediction': y_pred_results, 'Weight':accuracy_scores})

    predict_table['Weight']=predict_table['Weight']/predict_table['Weight'].sum()

    final_result=''
    num = (predict_table['Prediction']*predict_table['Weight']).sum()
    if num > 0.5:
        final_result='퇴사'
    else:
        final_result='생존'
    
    # print(f'{num*100}% 확률로 {final_result}')
    
    return predict_table
