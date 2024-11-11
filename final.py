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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(page_title="Fraud Detection", page_icon="🕵️‍♂️", layout="wide")

# Función para cargar datos
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("kartik2112/fraud-detection")
    csv_1 = os.path.join(path, 'fraudTest.csv')
    csv_2 = os.path.join(path, 'fraudTrain.csv')
    data_test = pd.read_csv(csv_1)
    data_train = pd.read_csv(csv_2)
    subsample_ratio = 0.1
    data_test_subsampled = resample(data_test, replace=False, n_samples=int(len(data_test) * subsample_ratio), random_state=42)
    data_train_subsampled = resample(data_train, replace=False, n_samples=int(len(data_train) * subsample_ratio), random_state=42)
    data_test_subsampled.to_csv('fraudTest2.csv', index=False)
    data_train_subsampled.to_csv('fraudTrain2.csv', index=False)
    data_test = pd.read_csv('fraudTest2.csv')
    data_train = pd.read_csv("fraudTrain2.csv")
    data = pd.concat([data_train, data_test])
    return data

# Función para preparar datos
@st.cache_data
def prepare_data(data):
    fraud = data[data["is_fraud"] == 1]
    not_fraud = data[data["is_fraud"] == 0]
    not_fraud = not_fraud.sample(fraud.shape[0])
    data = pd.concat([fraud, not_fraud])

    categorical_cols = ['merchant', 'category', 'gender', 'state']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    data = data.drop(['Unnamed: 0', 'first', 'last', 'unix_time', 'street', 'gender', 'job', 'dob', 'city', 'state', 'trans_num', 'merchant', 'cc_num'], axis=1)
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['trans_day'] = data['trans_date_trans_time'].dt.day
    data['trans_month'] = data['trans_date_trans_time'].dt.month
    data['trans_year'] = data['trans_date_trans_time'].dt.year
    data['trans_hour'] = data['trans_date_trans_time'].dt.hour
    data['trans_minute'] = data['trans_date_trans_time'].dt.minute
    data.drop(columns=['trans_date_trans_time'], inplace=True)

    encoder = LabelEncoder()
    data['category'] = encoder.fit_transform(data['category'])
    scaler = StandardScaler()
    data['amt'] = scaler.fit_transform(data[['amt']])
    data['zip'] = scaler.fit_transform(data[['zip']])
    data['city_pop'] = scaler.fit_transform(data[['city_pop']])
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud'].values  # Convertir a array de NumPy
    return X, y

# Función para entrenar modelos
@st.cache_data
def train_models(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    xgb_model = XGBClassifier(eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    return lr_model, rf_model, xgb_model

# Cargar y preparar los datos
data = load_data()
X, y = prepare_data(data)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar los modelos
lr_model, rf_model, xgb_model = train_models(X_train, y_train)

# Realizar predicciones y calcular métricas para el modelo de regresión logística
y_pred_lr = lr_model.predict(X_test)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]
cm_lr = confusion_matrix(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_proba_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_proba_lr)
report_lr = classification_report(y_test, y_pred_lr, output_dict=True)
report_df_lr = pd.DataFrame(report_lr).transpose()

# Realizar predicciones y calcular métricas para el modelo de Random Forest
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
cm_rf = confusion_matrix(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_proba_rf)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
report_df_rf = pd.DataFrame(report_rf).transpose()

# Realizar preddiciones y calcular métricas para el modelo XGBoost
y_pred_xgb = xgb_model.predict(X_test)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_proba_xgb)
report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
report_df_xgb = pd.DataFrame(report_xgb).transpose()

# Crear gráficos para la regresión logística
cm_df_lr = pd.DataFrame(cm_lr, index=['No Fraude', 'Fraude'], columns=['No Fraude', 'Fraude']).reset_index().melt(id_vars='index')
cm_df_lr.columns = ['Real', 'Predicción', 'Valor']
heatmap_lr = alt.Chart(cm_df_lr).mark_rect().encode(
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
roc_df_lr = pd.DataFrame({'FPR': fpr_lr, 'TPR': tpr_lr, 'Thresholds': thresholds_lr})
roc_curve_chart_lr = alt.Chart(roc_df_lr).mark_line().encode(
    x=alt.X('FPR', title='Falsos positivos'),
    y=alt.Y('TPR', title='Verdaderos positivos'),
    tooltip=['FPR', 'TPR', 'Thresholds']
)
diagonal_lr = alt.Chart(pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})).mark_line(strokeDash=[5, 5], color='gray').encode(
    x='FPR',
    y='TPR'
)
combined_chart_lr = alt.layer(roc_curve_chart_lr, diagonal_lr).properties(
    width=600,
    height=400,
    title=f'Curva ROC - Regresión Logística (AUC = {roc_auc_lr:.2f})'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Crear gráficos para Random Forest
cm_df_rf = pd.DataFrame(cm_rf, index=['No Fraude', 'Fraude'], columns=['No Fraude', 'Fraude']).reset_index().melt(id_vars='index')
cm_df_rf.columns = ['Real', 'Predicción', 'Valor']
heatmap_rf = alt.Chart(cm_df_rf).mark_rect().encode(
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
roc_df_rf = pd.DataFrame({'FPR': fpr_rf, 'TPR': tpr_rf, 'Thresholds': thresholds_rf})
roc_curve_chart_rf = alt.Chart(roc_df_rf).mark_line().encode(
    x=alt.X('FPR', title='Falsos positivos'),
    y=alt.Y('TPR', title='Verdaderos positivos'),
    tooltip=['FPR', 'TPR', 'Thresholds']
)
diagonal_rf = alt.Chart(pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})).mark_line(strokeDash=[5, 5], color='gray').encode(
    x='FPR',
    y='TPR'
)
combined_chart_rf = alt.layer(roc_curve_chart_rf, diagonal_rf).properties(
    width=600,
    height=400,
    title=f'Curva ROC - Random Forest (AUC = {roc_auc_rf:.2f})'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)
feature_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances_df = feature_importances.reset_index()
feature_importances_df.columns = ['Feature', 'Importance']
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# Crear gráficos para XGBoost
cm_df_xgb = pd.DataFrame(cm_xgb, index=['No Fraude', 'Fraude'], columns=['No Fraude', 'Fraude']).reset_index().melt(id_vars='index')
cm_df_xgb.columns = ['Real', 'Predicción', 'Valor']
heatmap_xgb = alt.Chart(cm_df_xgb).mark_rect().encode(
    x='Predicción:O',
    y='Real:O',
    color=alt.Color('Valor:Q', scale=alt.Scale(scheme='blues', reverse=True)),
    tooltip=['Real', 'Predicción', 'Valor']
).properties(
    width=400,
    height=300,
    title='Matriz de Confusión - XGBoost'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)
roc_df_xgb = pd.DataFrame({'FPR': fpr_xgb, 'TPR': tpr_xgb, 'Thresholds': thresholds_xgb})
roc_curve_chart_xgb = alt.Chart(roc_df_xgb).mark_line().encode(
    x=alt.X('FPR', title='Falsos positivos'),
    y=alt.Y('TPR', title='Verdaderos positivos'),
    tooltip=['FPR', 'TPR', 'Thresholds']
)
diagonal_xgb = alt.Chart(pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1]})).mark_line(strokeDash=[5, 5], color='gray').encode(
    x='FPR',
    y='TPR'
)
combined_chart_xgb = alt.layer(roc_curve_chart_xgb, diagonal_xgb).properties(
    width=600,
    height=400,
    title=f'Curva ROC - XGBoost (AUC = {roc_auc_xgb:.2f})'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)
feature_importances_xgb = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances_df_xgb = feature_importances_xgb.reset_index()
feature_importances_df_xgb.columns = ['Feature', 'Importance']
feature_importances_df_xgb = feature_importances_df_xgb.sort_values(by='Importance', ascending=False)


# Sidebar para seleccionar el modelo
model_choice = st.sidebar.selectbox("Selecciona el modelo", ["Regresión Logística", "Random Forest", "XGBoost"])

# Mostrar las gráficas según el modelo seleccionado
if model_choice == "Regresión Logística":
    st.subheader('Regresión Logística')
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(report_df_lr)
        with col2:
            st.altair_chart(heatmap_lr, use_container_width=True)
        with col3:
            st.altair_chart(combined_chart_lr, use_container_width=True)
elif model_choice == "Random Forest":
    st.subheader('Random Forest')
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(report_df_rf)
        with col2:
            st.altair_chart(heatmap_rf, use_container_width=True)
        with col3:
            st.altair_chart(combined_chart_rf, use_container_width=True)
    st.bar_chart(feature_importances_df.set_index('Feature'))
elif model_choice == "XGBoost":
    st.subheader('XGBoost')
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.dataframe(report_df_xgb)
        with col2:
            st.altair_chart(heatmap_xgb, use_container_width=True)
        with col3:
            st.altair_chart(combined_chart_xgb, use_container_width=True)
    st.bar_chart(feature_importances_df_xgb.set_index('Feature'))