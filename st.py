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
chart = alt.Chart(data).mark_bar(color='skyblue', size=1).encode(
    x=alt.X('cc_num', title='Número de Tarjeta de Crédito'),
    y=alt.Y('frequency', title='Cantidad de Transacciones')
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

cc_frequency_df = df['cc_num'].value_counts().reset_index()
cc_frequency_df.columns = ['cc_num', 'frequency']
filtered_cc_frequency_df = cc_frequency_df[cc_frequency_df['cc_num'].astype(float) > 1*10**18]
data = filtered_cc_frequency_df

# Gráfico con Altair
chart = alt.Chart(data).mark_bar(color='skyblue', size=1).encode(
    x=alt.X('cc_num', title='Número de Tarjeta de Crédito'),
    y=alt.Y('frequency', title='Cantidad de Transacciones')
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

fraud_transactions = df[df['is_fraud'] == 1]

# Contar la cantidad de transacciones fraudulentas por tarjeta
fraud_count_by_card = fraud_transactions['cc_num'].value_counts().reset_index()
fraud_count_by_card.columns = ['cc_num', 'frequency']
fraud_count_by_card = fraud_count_by_card[fraud_count_by_card['cc_num'].astype(float) < 1*10**18]
data = fraud_count_by_card

# Gráfico con Altair
chart = alt.Chart(data).mark_bar(color='skyblue', size=1).encode(
    x=alt.X('cc_num', title='Número de Tarjeta de Crédito'),
    y=alt.Y('frequency', title='Cantidad de Transacciones Fraudulentas')
).properties(
    width=600,
    height=400,
    title='Cantidad de Transacciones Fraudulentas por Tarjeta de Crédito'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar gráfico en Streamlit
st.altair_chart(chart, use_container_width=True)


fraud_transactions = df[df['is_fraud'] == 1]

# Contar la cantidad de transacciones fraudulentas por tarjeta
fraud_count_by_card = fraud_transactions['cc_num'].value_counts().reset_index()
fraud_count_by_card.columns = ['cc_num', 'frequency']
fraud_count_by_card = fraud_count_by_card[fraud_count_by_card['cc_num'].astype(float) > 1*10**18]
data = fraud_count_by_card

# Gráfico con Altair
chart = alt.Chart(data).mark_bar(color='skyblue', size=1).encode(
    x=alt.X('cc_num', title='Número de Tarjeta de Crédito'),
    y=alt.Y('frequency', title='Cantidad de Transacciones Fraudulentas')
).properties(
    width=600,
    height=400,
    title='Cantidad de Transacciones Fraudulentas por Tarjeta de Crédito'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar gráfico en Streamlit
st.altair_chart(chart, use_container_width=True)

df=df.drop('cc_num',axis=1)
df=df.drop('first',axis=1)
df=df.drop('last',axis=1)
df=df.drop('street',axis=1)
df=df.drop('city',axis=1)
df=df.drop('zip',axis=1)

st.divider()
st.subheader("Cantidad de Fraudes por Mes")

# cantidad de fraudes por mes
fraud_df = df[df['is_fraud'] == 1]

# Agrupar por mes para contar los fraudes por mes
fraud_df['month'] = fraud_df['trans_date_trans_time'].dt.to_period('M')
fraud_count_by_month = fraud_df.groupby('month').size().reset_index(name='frequency')

# Convertir la columna 'month' a cadena de texto
fraud_count_by_month['month'] = fraud_count_by_month['month'].astype(str)

# Gráfico con Altair
chart = alt.Chart(fraud_count_by_month).mark_bar(color='skyblue', size=15).encode(
    x=alt.X('month', title='Mes', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('frequency', title='Cantidad de Fraudes')
).properties(
    width=600,
    height=400,
).configure_axis(
    labelFontSize=12,
    titleFontSize=14
).configure_title(
    fontSize=16
)

# Mostrar gráfico en Streamlit
st.altair_chart(chart, use_container_width=True)