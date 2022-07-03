import pandas as pd
import numpy as np

def new_e(new_emp) :
    # list 를 df로 변환
    # new_emp = pd.DataFrame([new_list], columns = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'])
    # 학사, 석박사 구분
    new_emp.loc[new_emp['Education'] != "Bachelors", 'edu'] = 'M&D'
    # 파생변수들 생성하기
    new_emp['age_gr'] = new_emp['Age'].copy()
    new_emp['age_log'] = new_emp['Age'].copy()
    # PaymentTier string type으로ㅣ
    new_emp.PaymentTier = new_emp.PaymentTier.astype('str')
    # 나이 범주 바꿔주기
    new_emp.loc[((new_emp['Age'] >= 20) & (new_emp['Age'] < 30)), 'age_gr'] = '20'
    new_emp.loc[((new_emp['Age'] >= 30) & (new_emp['Age'] < 40)), 'age_gr'] = '30'
    new_emp.loc[((new_emp['Age'] >= 40) & (new_emp['Age'] < 50)), 'age_gr'] = '40'
    new_emp['age_log'] = np.log1p(new_emp['age_log'])
    return new_emp

def new_t(blah) :
    new_df = new_e(blah)
    f = ['JoiningYear', 'City', 'PaymentTier', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain', 'edu', 'age_gr', 'age_log']
    new_df = new_df[f]
    tmp = pd.DataFrame(columns = ['JoiningYear', 'ExperienceInCurrentDomain', 'age_log', 'City_Bangalore',
       'City_New Delhi', 'City_Pune', 'PaymentTier_1', 'PaymentTier_2',
       'PaymentTier_3', 'Gender_Female', 'Gender_Male', 'EverBenched_No',
       'EverBenched_Yes', 'edu_Bachelors', 'edu_M&D', 'age_gr_20', 'age_gr_30',
       'age_gr_40'])
    tmp[['JoiningYear', 'ExperienceInCurrentDomain', 'age_log']] = new_df[['JoiningYear', 'ExperienceInCurrentDomain', 'age_log']]

    # City 변환
    if (new_df['City'] == 'Bangalore').bool() : 
        tmp.loc[0,'City_Bangalore'] = 1
    elif (new_df['City'] == 'New Delhi').bool() : 
        tmp.loc[0,'City_New Delhi'] = 1
    elif (new_df['City'] == 'Pune').bool() : 
        tmp.loc[0,'City_Pune'] = 1
    # PaymentTier 변환
    if (new_df['PaymentTier'] == 1).bool() : 
        tmp.loc[0,'PaymentTier_1'] = 1
    elif (new_df['PaymentTier'] == '2').bool() : 
        tmp.loc[0,'PaymentTier_1'] = 1
    elif (new_df['PaymentTier'] == '3').bool() : 
        tmp.loc[0,'PaymentTier_1'] = 1
    # Gender 변환
    if (new_df['Gender'] == 'Female').bool() : 
        tmp.loc[0,'Gender_Female'] = 1
    elif (new_df['Gender'] == 'Male').bool() : 
        tmp.loc[0,'Gender_Male'] = 1
    # EverBenched 변환
    if (new_df['EverBenched'] == 'Yes').bool() : 
        tmp.loc[0,'EverBenched_Yes'] = 1
    elif (new_df['EverBenched'] == 'No').bool() : 
        tmp.loc[0,'EverBenched_No'] = 1
    # edu 변환
    if (new_df['edu'] == 'Bachelors').bool() : 
        tmp.loc[0,'edu_Bachelors'] = 1
    elif (new_df['edu'] == 'M&D').bool() : 
        tmp.loc[0,'edu_M&D'] = 1
    # age_gr 변환
    if (new_df['age_gr'] == '20').bool() : 
        tmp.loc[0,'age_gr_20'] = 1
    elif (new_df['age_gr'] == '30').bool() : 
        tmp.loc[0,'age_gr_30'] = 1
    elif (new_df['age_gr'] == '40').bool() : 
        tmp.loc[0,'age_gr_40'] = 1
        
    # fillna
    tmp = tmp.fillna(value = 0)
    return tmp