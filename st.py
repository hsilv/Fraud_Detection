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
from pycaret.classification import *

# descargar la última versión del archivo
path = kagglehub.dataset_download("kartik2112/fraud-detection")

st.set_page_config(page_title="Análisis Exploratorio de Datos", page_icon=":bar_chart:", layout="centered")
st.title("Análisis Exploratorio de Datos")
csv_file_path = os.path.join(path, 'fraudTrain.csv')
# cargar el dataset en un DataFrame
df = pd.read_csv(csv_file_path, index_col=0)

st.divider()

st.subheader("Tabla de datos (primeras 10 filas)")
st.dataframe(df.head(10))

df = df.dropna()
df = df.drop_duplicates()
df=df.drop('trans_num',axis=1)
df=df.drop('unix_time',axis=1)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])

st.divider()
st.subheader("Dataset Limpio")
st.dataframe(df.head(10))

st.divider()
st.subheader("Distribución de transacciones fraudulentas y no fraudulentas")
fraudulent_count = df[df['is_fraud'] == 1].shape[0]
non_fraudulent_count = df[df['is_fraud'] == 0].shape[0]
fig, ax = plt.subplots()
sns.countplot(x='is_fraud', data=df, ax=ax)
ax.set_title("Distribución de transacciones fraudulentas y no fraudulentas")
ax.set_xticklabels(['No Fraudulenta', 'Fraudulenta'])
st.bar_chart(df['is_fraud'].value_counts())

st.markdown(f"**Distribución de clases (en porcentaje):**")
distribucion = df['is_fraud'].value_counts(normalize=True) * 100
st.table(distribucion)

st.divider()
st.subheader("Cantidad de transacciones por Tarjeta de Crédito")
cc_frequency_df = df['cc_num'].value_counts().reset_index()
cc_frequency_df.columns = ['cc_num', 'frequency']
filtered_cc_frequency_df = cc_frequency_df[cc_frequency_df['cc_num'].astype(float) < 1*10**18]
data = filtered_cc_frequency_df

# Gráfico con Altair
chart = alt.Chart(data).mark_bar(color='skyblue', stroke='black').encode(
    x='cc_num',
    y='frequency'
).properties(
    width=600,
    height=400,
    title='Cantidad de Transacciones por Tarjeta de Crédito'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar gráfico en Streamlit
st.altair_chart(chart, use_container_width=True)