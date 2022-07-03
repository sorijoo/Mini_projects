import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


st.set_page_config(layout="centered")

st.title("MAKE YOUR OWN MACHINE LEARNING MODEL!")

st.write("""
# Employee Future Prediction
this app predict Employee's Future
""")


def ConvertEducation(x):
    if x == "Bachelors":
        return 1
    elif x == "Masters":
        return 2
    else :
        return 3         

def ConvertCity(x):
    if x == "Bangalore":
        return 1
    elif x == "Pune":
        return 2
    else :
        return 3    


def scoreModel(model, X_train, X_valid, y_train, y_valid):
    '''
        머신러닝 모델과 X_train, X_valid, y_train, y_valid 변수를 받아서
        모델명, 학습용 세트 정확도(R2 score), 테스트 세트 정확도(R2 score)를 출력하는 함수
    '''
    print("모델 : {}".format(model))
    print("학습용 세트 정확도: {:.3f}".format(model.score(X_train, y_train)))
    valid_score = model.score(X_valid, y_valid)
    print("검증 세트 정확도: {:.3f}".format(valid_score))
    return valid_score        





df = pd.read_csv("data/Employee.csv")



#데이터 전처리

df["Gender_n"] = df["Gender"].apply(lambda x : int(x=="Female"))
df["EverBenched_n"] = df["EverBenched"].apply(lambda x : int(x=="Yes"))
df["Education_n"] = df["Education"].map(lambda x :ConvertEducation(x))
df["City_n"] = df["City"].map(lambda x :ConvertCity(x))


#예측 데이터 정하기

label_name = "LeaveOrNot"
feature_names = df.columns.tolist()

no = [label_name, 'Education', 'City', 'Gender','EverBenched']
for i in no:
    feature_names.remove(i) 

X = df[feature_names]
y = df[label_name]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)



# st.write(X_train)
# st.write(X_test)


## 모델 선택 - 로지스틱 리그레션

st.write("## Logistic Regression")
col1, col2 = st.columns(2)
with col1:
  
    penalty = st.selectbox('Penalty', ('l2', 'none'))
    st.markdown("<br/>", unsafe_allow_html=True)
    c = float(st.slider('C', 0.0, 100.0, 10.0))

with col2:
    solver = st.selectbox('Solver', ('newton-cg','lbfgs', 'liblinear','sag','saga'))
    st.markdown("<br/>", unsafe_allow_html=True)
    max_iter = float(st.slider('Max iter', 500.0, 10000.0, 7000.0))
 

st.markdown("<br/>", unsafe_allow_html=True)

model_lr = st.button('Logistic Regression 모델링 Start')

if model_lr:
 
    lr = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=max_iter)

    lr_model_state = st.text('인생을 예측하는 데에는 시간이 걸립니다')
    lr_model = lr.fit(X_train, y_train)

    lr_model_state.success("완료!")

    y_predict = lr_model.predict(X_test)
    score = scoreModel(lr_model, X_train, X_test, y_train, y_test)
    st.metric("이 모델의 정확도는 : ", round(score,5))
    if score < 0.7 :
        st.image("img/cat_meme.jpeg")
    else:
        st.image("img/ai.jpeg")    
   

## 모델 선택 - 랜덤 포레스트

st.write("## Random Forest")
col1, col2 = st.columns(2)
with col1:
    n_estimators = int(st.slider('N estimators', 5, 500, 300))
    st.markdown("<br/>", unsafe_allow_html=True)
    min_samples_split = int(st.slider('Min samples split', 1, 30, 10))
    st.markdown("<br/>", unsafe_allow_html=True)
    min_samples_leaf = float(st.slider('Min samples leaf', 0.0, 0.5, 0.3))


with col2:
    max_depth= int(st.slider('Max depth', 1, 10, 3))
    st.markdown("<br/>", unsafe_allow_html=True)
    max_leaf_nodes= int(st.slider('Max leaf nodes', 1, 10, 3))
 

st.markdown("<br/>", unsafe_allow_html=True)

model_rf = st.button('Random Forest 모델링 Start')

if model_rf:

    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

    rf_model_state = st.text('인생을 예측하는 데에는 시간이 걸립니다')
    rf_model = rf.fit(X_train, y_train)

    rf_model_state.success("완료!")

    y_predict = rf_model.predict(X_test)
    score = scoreModel(rf_model, X_train, X_test, y_train, y_test)
    st.metric("이 모델의 정확도는 : ", round(score,5))
    if score < 0.7 :
        st.image("img/cat_meme.jpeg")
    else:
        st.image("img/ai.jpeg")    



#세번째 모델

st.write("## K-Nearest Neighbors")
col1, col2 = st.columns(2)
with col1:
    n_neighbors = int(st.slider('N neighbors', 2, 30, 5))
    st.markdown("<br/>", unsafe_allow_html=True)
    weights = st.selectbox('Weight', ('uniform', 'distance'))
    


with col2:
    leaf_size= int(st.slider('Max depth', 1, 100, 30))
    st.markdown("<br/>", unsafe_allow_html=True)
    algorithm = st.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))
 

st.markdown("<br/>", unsafe_allow_html=True)

model_knn = st.button('K-Nearest Neighbors 모델링 Start')

if model_knn:
    
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size, algorithm=algorithm)
    knn_model_state = st.text('인생을 예측하는 데에는 시간이 걸립니다')
    knn_model = knn.fit(X_train, y_train)
    knn_model_state.success("완료!")
    y_predict = knn_model.predict(X_test)
    score = scoreModel(knn_model, X_train, X_test, y_train, y_test)
    st.metric("이 모델의 정확도는 : ", round(score,5))
    if score < 0.7 :
        st.image("img/cat_meme.jpeg")
    else:
        st.image("img/ai.jpeg")    


# clf=RandomForestClassifier()
# clf.fit(X,Y)

# y_predict = lr.fit(X_train, y_train).predict(X_test)
# # prediction_proba=clf.predict_proba(df)

# st.subheader('class label and there corresponding index number ')
# st.write(iris.target_names)

# st.subheader('prediction ')
# st.write(iris.target_names[prediction])

# st.subheader('prediction probability ')
# st.write(prediction_proba)
