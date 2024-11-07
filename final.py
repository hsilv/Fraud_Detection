import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
plt.rcParams['font.family'] = 'Liberation Sans'
import seaborn as sns
import kagglehub
import os
import pycaret
from sklearn.utils import resample
from pycaret.classification import *
import warnings
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Detection", page_icon="🕵️‍♂️", layout="centered")
path = kagglehub.dataset_download("kartik2112/fraud-detection")
csv_1= os.path.join(path, 'fraudTest.csv')
csv_2= os.path.join(path, 'fraudTrain.csv')
data_test = pd.read_csv(csv_1)
data_train = pd.read_csv(csv_2)
subsample_ratio = 0.1
data_test_subsampled = resample(data_test, replace=False, n_samples=int(len(data_test) * subsample_ratio), random_state=42)
data_train_subsampled = resample(data_train, replace=False, n_samples=int(len(data_train) * subsample_ratio), random_state=42)
data_test_subsampled.to_csv('fraudTest2.csv', index=False)
data_train_subsampled.to_csv('fraudTrain2.csv', index=False)

data_test = pd.read_csv('fraudTest2.csv')
data_train = pd.read_csv("fraudTrain2.csv")
data = pd.concat([data_train,data_test])

fraud=data[data["is_fraud"]==1]
not_fraud=data[data["is_fraud"]==0]
not_fraud=not_fraud.sample(fraud.shape[0])
data=pd.concat([fraud,not_fraud])

st.title("Fraud Detection")
st.dataframe(data.tail())


categorical_cols = ['merchant', 'category', 'gender', 'state']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le


data = data.drop(['Unnamed: 0','first','last','unix_time','street','gender','job','dob','city','state','trans_num','merchant','cc_num'], axis=1)

data['trans_date_trans_time']=pd.to_datetime(data['trans_date_trans_time'])
data['trans_day']=data['trans_date_trans_time'].dt.day
data['trans_month']=data['trans_date_trans_time'].dt.month
data['trans_year']=data['trans_date_trans_time'].dt.year
data['trans_hour']=data['trans_date_trans_time'].dt.hour
data['trans_minute']=data['trans_date_trans_time'].dt.minute
data.drop(columns=['trans_date_trans_time'],inplace=True)

encoder=LabelEncoder()
data['category']=encoder.fit_transform(data['category'])
#data['cc_num']=encoder.fit_transform(data['cc_num'])
scaler=StandardScaler()
data['amt']=scaler.fit_transform(data[['amt']])
data['zip']=scaler.fit_transform(data[['zip']])
data['city_pop']=scaler.fit_transform(data[['city_pop']])
#data['cc_num']=encoder.fit_transform(data['cc_num'])
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


st.write('y_train distribution:\n', y_train.value_counts())
st.write('y_test distribution:\n', y_test.value_counts())

# Entrenar el modelo de regresión logística
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Mostrar el informe de clasificación
st.divider()
st.subheader('Regresión logística')
report = classification_report(y_test, y_pred_lr, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Calcular la matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Convertir la matriz de confusión en un DataFrame
cm_df = pd.DataFrame(cm_lr, index=['No Fraude', 'Fraude'], columns=['Predicción No Fraude', 'Predicción Fraude'])

# Convertir el DataFrame en un formato largo para Altair
cm_df = cm_df.reset_index().melt(id_vars='index')
cm_df.columns = ['Real', 'Predicción', 'Valor']

# Crear el heatmap con Altair
heatmap = alt.Chart(cm_df).mark_rect().encode(
    x='Predicción:O',
    y='Real:O',
    color=alt.Color('Valor:Q', scale=alt.Scale(scheme='blues', reverse=True)),
    tooltip=['Real', 'Predicción', 'Valor']
).properties(
    width=400,
    height=300,
    title='Matriz de Confusión - Regresión Logística'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar el heatmap en Streamlit
st.altair_chart(heatmap, use_container_width=True)

# Calcular la curva ROC
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba_lr)

# Crear un DataFrame con los valores de la curva ROC
roc_df = pd.DataFrame({
    'FPR': fpr_lr,
    'TPR': tpr_lr,
    'Thresholds': thresholds_lr
})

# Crear la curva ROC con Altair
roc_curve_chart = alt.Chart(roc_df).mark_line().encode(
    x=alt.X('FPR', title='Falsos positivos'),
    y=alt.Y('TPR', title='Verdaderos positivos'),
    tooltip=['FPR', 'TPR', 'Thresholds']
)

# Crear la línea diagonal punteada
diagonal = alt.Chart(pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})).mark_line(strokeDash=[5, 5], color='gray').encode(
    x='FPR',
    y='TPR'
)

# Combinar la curva ROC y la línea diagonal
combined_chart = alt.layer(roc_curve_chart, diagonal).properties(
    width=600,
    height=400,
    title=f'Curva ROC - Regresión Logística (AUC = {roc_auc_lr:.2f})'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar la curva ROC en Streamlit
st.altair_chart(combined_chart, use_container_width=True)

st.divider()
st.subheader('Random Forest')
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)


rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred_rf, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Calcular la matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_rf)

# Convertir la matriz de confusión en un DataFrame
cm_df = pd.DataFrame(cm_lr, index=['No Fraude', 'Fraude'], columns=['Predicción No Fraude', 'Predicción Fraude'])

# Convertir el DataFrame en un formato largo para Altair
cm_df = cm_df.reset_index().melt(id_vars='index')
cm_df.columns = ['Real', 'Predicción', 'Valor']

# Crear el heatmap con Altair
heatmap = alt.Chart(cm_df).mark_rect().encode(
    x='Predicción:O',
    y='Real:O',
    color=alt.Color('Valor:Q', scale=alt.Scale(scheme='blues', reverse=True)),
    tooltip=['Real', 'Predicción', 'Valor']
).properties(
    width=400,
    height=300,
    title='Matriz de Confusión - Random Forest'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar el heatmap en Streamlit
st.altair_chart(heatmap, use_container_width=True)

# Calcular la curva ROC
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)

# Crear un DataFrame con los valores de la curva ROC
roc_df = pd.DataFrame({
    'FPR': fpr_rf,
    'TPR': tpr_rf,
    'Thresholds': thresholds_rf
})

# Crear la curva ROC con Altair
roc_curve_chart = alt.Chart(roc_df).mark_line().encode(
    x=alt.X('FPR', title='Falsos positivos'),
    y=alt.Y('TPR', title='Verdaderos positivos'),
    tooltip=['FPR', 'TPR', 'Thresholds']
)

# Crear la línea diagonal punteada
diagonal = alt.Chart(pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})).mark_line(strokeDash=[5, 5], color='gray').encode(
    x='FPR',
    y='TPR'
)

# Combinar la curva ROC y la línea diagonal
combined_chart = alt.layer(roc_curve_chart, diagonal).properties(
    width=600,
    height=400,
    title=f'Curva ROC - Random Forest (AUC = {roc_auc_rf:.2f})'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar la curva ROC en Streamlit
st.altair_chart(combined_chart, use_container_width=True)